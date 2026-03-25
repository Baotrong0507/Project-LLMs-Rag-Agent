import streamlit as st
import logging
import os
import tempfile
import json
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("smartdoc.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# LangChain imports
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Cross-encoder for re-ranking (Câu 9)
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except:
    CROSS_ENCODER_AVAILABLE = False

# ========================
# CẤU HÌNH TRANG
# ========================
st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="📄",
    layout="wide"
)

# ========================
# CSS
# ========================
st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .answer-box {
        background-color: #ffffff;
        color: #212529;
        border-left: 4px solid #007BFF;
        padding: 16px;
        border-radius: 8px;
        margin-top: 12px;
        font-size: 15px;
        line-height: 1.6;
    }
    .chat-user {
        background-color: #007BFF;
        color: white;
        padding: 10px 14px;
        border-radius: 12px 12px 0 12px;
        margin: 6px 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 14px;
    }
    .chat-bot {
        background-color: #F0F2F6;
        color: #212529;
        padding: 10px 14px;
        border-radius: 12px 12px 12px 0;
        margin: 6px 0;
        max-width: 80%;
        font-size: 14px;
    }
    .citation-box {
        background-color: #FFF3CD;
        color: #212529;
        border-left: 3px solid #FFC107;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 13px;
        margin: 4px 0;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover { background-color: #0056b3; }
    .doc-tag {
        background-color: #E8F4FD;
        color: #0066CC;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin: 2px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# SESSION STATE INIT
# ========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []        # Câu 2: Lưu lịch sử
if "documents_store" not in st.session_state:
    st.session_state.documents_store = {}     # Câu 8: Multi-document
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []

# ========================
# LOAD EMBEDDER & CROSS-ENCODER
# ========================
@st.cache_resource(show_spinner=False)
def load_embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource(show_spinner=False)
def load_cross_encoder():
    if CROSS_ENCODER_AVAILABLE:
        try:
            return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except:
            return None
    return None

# ========================
# CÂU 1: LOAD PDF VÀ DOCX
# ========================
def load_document(file_bytes, filename):
    logger.info(f"Loading file: {filename}")	
    ext = filename.lower().split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        if ext == "pdf":
            loader = PDFPlumberLoader(tmp_path)
        elif ext == "docx":
            loader = Docx2txtLoader(tmp_path)
        else:
            raise ValueError(f"Định dạng '{ext}' không được hỗ trợ!")
        docs = loader.load()
        # Câu 8: Gắn metadata
        for doc in docs:
            doc.metadata["source_file"] = filename
            doc.metadata["file_type"] = ext
            doc.metadata["upload_time"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        return docs
    finally:
        os.unlink(tmp_path)

# ========================
# CÂU 4: CHUNK STRATEGY
# ========================
def split_documents(docs, strategy, chunk_size, chunk_overlap):
    if strategy == "Token-based":
        splitter = TokenTextSplitter(
            chunk_size=chunk_size // 4,
            chunk_overlap=chunk_overlap // 4
        )
    elif strategy == "Paragraph-based":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n"]
        )
    else:  # Recursive (Mặc định)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
    chunks = splitter.split_documents(docs)

    logger.info(f"Split thành {len(chunks)} chunks")

    return chunks

# ========================
# CÂU 7: HYBRID SEARCH
# ========================
def build_retriever(chunks, embedder, search_mode, top_k):
    vector_store = FAISS.from_documents(chunks, embedder)
    if search_mode == "Hybrid (Vector + BM25)":
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = top_k
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )
    elif search_mode == "MMR (Đa dạng)":
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": top_k * 3, "lambda_mult": 0.7}
        )
    else:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
    return retriever, vector_store

# ========================
# CÂU 9: RE-RANKING
# ========================
def rerank_documents(query, docs, cross_encoder, top_k):
    if cross_encoder is None or len(docs) == 0:
        return docs
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]

# ========================
# PHÁT HIỆN NGÔN NGỮ
# ========================
def detect_language(text):
    vi_chars = 'àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ'
    count = sum(1 for c in text.lower() if c in vi_chars)
    return 'vi' if count > 2 else 'en'

# ========================
# CÂU 10: SELF-RAG EVALUATION
# ========================
def self_rag_evaluate(llm, question, answer, context):
    eval_prompt = f"""Đánh giá câu trả lời dựa trên ngữ cảnh.
Trả lời CHỈ bằng JSON: {{"score": 1-10, "reason": "lý do", "is_sufficient": true/false}}

Ngữ cảnh: {context[:400]}
Câu hỏi: {question}
Câu trả lời: {answer}

JSON:"""
    try:
        result = llm.invoke(eval_prompt)
        start = result.find("{")
        end = result.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(result[start:end])
    except:
        pass
    return {"score": 7, "reason": "Không thể đánh giá tự động", "is_sufficient": True}

# ========================
# CÂU 10: QUERY REWRITING
# ========================
def rewrite_query(llm, question, chat_history):
    if not chat_history:
        return question
    history_text = "\n".join([
        f"User: {h['question']}\nBot: {h['answer'][:80]}"
        for h in chat_history[-2:]
    ])
    prompt = f"""Dựa vào lịch sử, viết lại câu hỏi cho rõ ràng hơn.
Chỉ trả lời câu hỏi đã viết lại.

Lịch sử:
{history_text}

Câu hỏi gốc: {question}
Câu hỏi viết lại:"""
    try:
        rewritten = llm.invoke(prompt).strip()
        if 5 < len(rewritten) < 300:
            return rewritten
    except:
        pass
    return question

# ========================
# HÀM TRẢ LỜI CHÍNH
# ========================
def get_answer(question, retriever, use_rerank, use_self_rag,
               use_conversational, top_k, chat_history):
    logger.info(f"Query: {question}")

    llm = Ollama(model="qwen2.5:7b", temperature=0.7)
    lang = detect_language(question)
    cross_encoder = load_cross_encoder() if use_rerank else None

    # Câu 10: Query rewriting
    rewritten_q = question
    if use_conversational and chat_history:
        rewritten_q = rewrite_query(llm, question, chat_history)

    # Lấy docs liên quan
    relevant_docs = retriever.get_relevant_documents(rewritten_q)

    # Câu 9: Re-ranking
    if use_rerank and cross_encoder:
        relevant_docs = rerank_documents(rewritten_q, relevant_docs, cross_encoder, top_k)

    logger.info(f"Retrieved {len(relevant_docs)} documents")

    # Câu 5: Citation tracking
    citations = []
    for i, doc in enumerate(relevant_docs):
        citations.append({
            "index": i + 1,
            "content": doc.page_content[:200],
            "source": doc.metadata.get("source_file", "Unknown"),
            "page": doc.metadata.get("page", "N/A"),
        })

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Câu 6: Conversational RAG
    history_text = ""
    if use_conversational and chat_history:
        history_text = "\n".join([
            f"User: {h['question']}\nBot: {h['answer']}"
            for h in chat_history[-3:]
        ])

    if lang == 'vi':
        prompt = f"""{'Lịch sử hội thoại:' + chr(10) + history_text + chr(10) if history_text else ''}Ngữ cảnh: {context}

Câu hỏi: {question}
Trả lời ngắn gọn bằng tiếng Việt (3-4 câu). Nếu không biết hãy nói không biết.
Trả lời:"""
    else:
        prompt = f"""{'Conversation history:' + chr(10) + history_text + chr(10) if history_text else ''}Context: {context}

Question: {question}
Answer concisely in English (3-4 sentences). If unsure, say so.
Answer:"""

    answer = llm.invoke(prompt)

    elapsed = 0  # nếu muốn log thời gian thì tính trước khi invoke
    logger.info(f"Answer generated in {elapsed:.1f}s")

    # Câu 10: Self-RAG
    self_eval = None
    if use_self_rag:
        self_eval = self_rag_evaluate(llm, question, answer, context)

    return answer.strip(), citations, self_eval, rewritten_q

# ========================
# SIDEBAR
# ========================
with st.sidebar:
    st.title("⚙️ Cài đặt")
    st.markdown("---")

    st.markdown("### 📐 Chunk Strategy (Câu 4)")
    chunk_strategy = st.selectbox(
        "Chiến lược:",
        ["Recursive (Mặc định)", "Token-based", "Paragraph-based"]
    )
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 300, 100, 50)

    st.markdown("---")
    st.markdown("### 🔍 Search Mode (Câu 7)")
    search_mode = st.selectbox(
        "Chế độ tìm kiếm:",
        ["Similarity (Mặc định)", "Hybrid (Vector + BM25)", "MMR (Đa dạng)"]
    )
    top_k = st.slider("Top K kết quả", 1, 10, 3)

    st.markdown("---")
    st.markdown("### 🚀 Tính năng nâng cao")
    use_rerank = st.checkbox("Re-ranking Cross-Encoder (Câu 9)", value=False)
    use_self_rag = st.checkbox("Self-RAG Evaluation (Câu 10)", value=False)
    use_conversational = st.checkbox("Conversational RAG (Câu 6)", value=True)

    st.markdown("---")
    st.markdown("### 🤖 Model Info")
    st.info("**LLM:** Qwen2.5:7b (Ollama)")
    st.info("**Embedding:** multilingual-mpnet (768d)")
    st.info("**Vector DB:** FAISS")

    st.markdown("---")
    st.markdown("### 📁 Tài liệu (Câu 8)")
    if st.session_state.documents_store:
        for fname, info in st.session_state.documents_store.items():
            st.markdown(f'<span class="doc-tag">📄 {fname}</span>', unsafe_allow_html=True)
            st.caption(f"{info['num_chunks']} chunks | {info['upload_time']}")
    else:
        st.caption("Chưa có tài liệu")

    st.markdown("---")
    st.markdown("### 💬 Lịch sử (Câu 2)")
    if st.session_state.chat_history:
        st.caption(f"{len(st.session_state.chat_history)} câu hỏi")
        for i, h in enumerate(st.session_state.chat_history[-5:]):
            with st.expander(f"Q{i+1}: {h['question'][:35]}..."):
                st.write(f"**Q:** {h['question']}")
                st.write(f"**A:** {h['answer'][:150]}...")
                st.caption(h['timestamp'])
    else:
        st.caption("Chưa có lịch sử")

    # Câu 3: Nút xóa
    st.markdown("---")
    st.markdown("### 🗑️ Xóa dữ liệu (Câu 3)")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Xóa chat", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Đã xóa!")
            st.rerun()
    with col2:
        if st.button("🗑️ Xóa docs", use_container_width=True):
            st.session_state.documents_store = {}
            st.session_state.all_chunks = []
            st.success("Đã xóa!")
            st.rerun()

# ========================
# MAIN UI
# ========================
st.title("📄 SmartDoc AI")
st.markdown("**Hệ thống hỏi đáp thông minh** — RAG + Qwen2.5 | OSSD Spring 2026")
st.markdown("---")

# Câu 1 + Câu 8: Upload nhiều file PDF/DOCX
st.subheader("📤 Upload tài liệu — PDF & DOCX (Câu 1 + Câu 8)")
uploaded_files = st.file_uploader(
    "Chọn file PDF hoặc DOCX (có thể chọn nhiều file cùng lúc)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    embedder = load_embedder()
    new_files = [f for f in uploaded_files
                 if f.name not in st.session_state.documents_store]

    if new_files:
        for uf in new_files:
            with st.spinner(f"⏳ Xử lý **{uf.name}**..."):
                try:
                    file_bytes = uf.read()
                    docs = load_document(file_bytes, uf.name)
                    chunks = split_documents(docs, chunk_strategy, chunk_size, chunk_overlap)
                    retriever, vector_store = build_retriever(chunks, embedder, search_mode, top_k)

                    st.session_state.documents_store[uf.name] = {
                        "chunks": chunks,
                        "retriever": retriever,
                        "num_chunks": len(chunks),
                        "upload_time": datetime.now().strftime("%H:%M %d/%m")
                    }
                    st.session_state.all_chunks.extend(chunks)
                    st.success(f"✅ **{uf.name}** — {len(chunks)} chunks")
                except Exception as e:
                    st.error(f"❌ Lỗi {uf.name}: {str(e)}")
    else:
        st.info(f"✅ {len(uploaded_files)} file đã sẵn sàng")

# Phần Q&A
if st.session_state.documents_store:
    st.markdown("---")

    # Câu 8: Filter theo document
    all_doc_names = list(st.session_state.documents_store.keys())
    if len(all_doc_names) > 1:
        st.subheader("🔎 Lọc tài liệu (Câu 8)")
        selected_docs = st.multiselect(
            "Tìm kiếm trong tài liệu nào? (bỏ trống = tất cả)",
            all_doc_names,
            default=all_doc_names
        )
    else:
        selected_docs = all_doc_names

    st.subheader("💬 Đặt câu hỏi")

    # Hiển thị lịch sử dạng bubble chat
    if st.session_state.chat_history:
        st.markdown("#### 📜 Lịch sử hội thoại")
        for h in st.session_state.chat_history:
            st.markdown(f'<div class="chat-user">🧑 {h["question"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bot">🤖 {h["answer"]}</div>', unsafe_allow_html=True)
        st.markdown("---")

    question = st.text_input(
        "Nhập câu hỏi:",
        placeholder="Ví dụ: Tài liệu này nói về gì?",
        key="q_input"
    )

    if question:
        # Gom chunks từ docs được chọn
        selected_chunks = []
        for doc_name in (selected_docs if selected_docs else all_doc_names):
            selected_chunks.extend(st.session_state.documents_store[doc_name]["chunks"])

        with st.spinner("🤔 Đang xử lý câu trả lời..."):
            try:
                embedder = load_embedder()
                retriever, _ = build_retriever(selected_chunks, embedder, search_mode, top_k)

                t0 = time.time()
                answer, citations, self_eval, rewritten_q = get_answer(
                    question, retriever,
                    use_rerank, use_self_rag, use_conversational,
                    top_k, st.session_state.chat_history
                )
                elapsed = time.time() - t0

                # Query rewriting info
                if rewritten_q != question:
                    st.caption(f"🔄 Câu hỏi được cải thiện (Câu 10): *{rewritten_q}*")

                # Câu trả lời
                st.subheader("💡 Câu trả lời")
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                st.caption(f"⏱️ {elapsed:.1f}s")

                # Câu 10: Self-RAG
                if use_self_rag and self_eval:
                    score = self_eval.get("score", 0)
                    icon = "🟢" if score >= 7 else "🟡" if score >= 5 else "🔴"
                    st.info(f"{icon} **Self-RAG (Câu 10):** Điểm {score}/10 — {self_eval.get('reason', '')}")

                # Câu 5: Citations
                with st.expander(f"📚 Nguồn tham khảo — {len(citations)} đoạn (Câu 5)"):
                    for c in citations:
                        st.markdown(f"""
                        <div class="citation-box">
                            <strong>[{c['index']}] 📄 {c['source']} — Trang {c['page']}</strong><br>
                            <em>"{c['content']}..."</em>
                        </div>
                        """, unsafe_allow_html=True)

                # Câu 2: Lưu vào lịch sử
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "timestamp": datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                })

            except Exception as e:
                if "connection" in str(e).lower() or "ollama" in str(e).lower():
                    st.error("❌ Ollama chưa chạy! Mở terminal khác và chạy: `ollama serve`")
                else:
                    st.error(f"❌ Lỗi: {str(e)}")
                    st.exception(e)
else:
    st.info("👆 Upload file PDF hoặc DOCX để bắt đầu!")

st.markdown("---")
st.caption("SmartDoc AI — Đại học Sài Gòn | OSSD Spring 2026 | Powered by LangChain + FAISS + Ollama")
