"""
app.py — Entry point của SmartDoc AI
Chỉ chứa luồng chính, mọi logic đã tách vào src/ và ui/

Cấu trúc:
    app.py              ← Entry point (file này)
    src/
        logger.py       ← Logging config (7.2.5)
        document_loader.py  ← Load PDF/DOCX (Câu 1)
        chunker.py      ← Chunk strategy (Câu 4)
        retriever.py    ← Hybrid search + Re-ranking (Câu 7, 9)
        rag_engine.py   ← RAG pipeline, Self-RAG (Câu 6, 10)
    ui/
        styles.py       ← CSS (chỉ sửa đây khi đổi giao diện)
        sidebar.py      ← Sidebar UI (Câu 2, 3, 4, 7, 8)
        components.py   ← Reusable components
"""
import streamlit as st
from datetime import datetime

# UI
from ui.styles     import load_styles
from ui.sidebar    import render_sidebar
from ui.components import (
    render_chat_history,
    render_answer,
    render_citations,
    render_self_rag_badge,
    render_empty_state,
    render_rewrite_notice,
)

# Logic
from src.document_loader import load_document
from src.chunker         import split_documents
from src.retriever       import load_embedder, build_retriever
from src.rag_engine      import get_answer

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="📄",
    layout="wide"
)

# ========================
# CSS
# ========================
load_styles()

# ========================
# SESSION STATE
# ========================
if "chat_history"    not in st.session_state:
    st.session_state.chat_history    = []   # Câu 2
if "documents_store" not in st.session_state:
    st.session_state.documents_store = {}   # Câu 8
if "all_chunks"      not in st.session_state:
    st.session_state.all_chunks      = []

# ========================
# SIDEBAR — trả về config
# ========================
config = render_sidebar()

chunk_strategy    = config["chunk_strategy"]
chunk_size        = config["chunk_size"]
chunk_overlap     = config["chunk_overlap"]
search_mode       = config["search_mode"]
top_k             = config["top_k"]
use_rerank        = config["use_rerank"]
use_self_rag      = config["use_self_rag"]
use_conversational = config["use_conversational"]

# ========================
# MAIN UI
# ========================
st.title("📄 SmartDoc AI")
st.markdown("**Hệ thống hỏi đáp thông minh** — RAG + Qwen2.5 | OSSD Spring 2026")
st.markdown("---")

# ── Câu 1 + 8: Upload PDF & DOCX (nhiều file)
st.subheader("📤 Upload tài liệu — PDF & DOCX")
uploaded_files = st.file_uploader(
    "Chọn file PDF hoặc DOCX (có thể chọn nhiều file cùng lúc)",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    key=st.session_state.get("uploader_key",0)
)

if uploaded_files:
    embedder  = load_embedder()
    new_files = [f for f in uploaded_files
                 if f.name not in st.session_state.documents_store]

    if new_files:
        for uf in new_files:
            with st.spinner(f"⏳ Đang xử lý **{uf.name}**..."):
                try:
                    file_bytes = uf.read()
                    docs   = load_document(file_bytes, uf.name)
                    chunks = split_documents(docs, chunk_strategy, chunk_size, chunk_overlap)

                    st.session_state.documents_store[uf.name] = {
                        "chunks":      chunks,
                        "num_chunks":  len(chunks),
                        "upload_time": datetime.now().strftime("%H:%M %d/%m")
                    }
                    st.session_state.all_chunks.extend(chunks)
                    st.success(f"✅ **{uf.name}** — {len(chunks)} chunks")
                    st.session_state["q_input"] = ""

                except Exception as e:
                    st.error(f"❌ Lỗi khi xử lý {uf.name}: {str(e)}")
    else:
        st.info("✅ Tất cả file đã được tải lên trước đó.")

# ── Q&A
if st.session_state.documents_store:
    st.markdown("---")

    # Câu 8: Filter theo document
    all_doc_names = list(st.session_state.documents_store.keys())
    selected_docs = all_doc_names
    if len(all_doc_names) > 1:
        st.subheader("🔎 Lọc tài liệu (Câu 8)")
        selected_docs = st.multiselect(
            "Tìm kiếm trong những tài liệu nào? (bỏ trống = tất cả)",
            all_doc_names,
            default=all_doc_names
        )

    st.subheader("💬 Đặt câu hỏi")

    # Câu 2: Hiển thị lịch sử
    render_chat_history(st.session_state.chat_history)

    question = st.text_input(
        "Nhập câu hỏi của bạn:",
        placeholder="Ví dụ: Tài liệu này nói về gì? Hoặc tóm tắt chương 2...",
        key="q_input"
    )

    if question:
        selected_chunks = []
        for doc_name in (selected_docs if selected_docs else all_doc_names):
            selected_chunks.extend(st.session_state.documents_store[doc_name]["chunks"])

        if not selected_chunks:
            st.warning("⚠️ Không có nội dung nào để tìm kiếm.")
        else:
            with st.spinner("🤔 Đang tìm kiếm và sinh câu trả lời..."):
                try:
                    embedder  = load_embedder()
                    # Lấy filename đầu tiên trong selected_docs
                    current_filename = selected_docs[0] if selected_docs else None
                    retriever = build_retriever(selected_chunks, embedder, search_mode, top_k, current_filename)

                    answer, citations, self_eval, rewritten_q, elapsed = get_answer(
                        question, retriever,
                        use_rerank, use_self_rag, use_conversational,
                        top_k, st.session_state.chat_history
                    )

                    # Câu 10: Query rewrite notice
                    render_rewrite_notice(question, rewritten_q)

                    # Answer
                    render_answer(answer, elapsed)

                    # Câu 10: Self-RAG badge
                    render_self_rag_badge(self_eval)

                    # Câu 5: Citations
                    render_citations(citations)

                    # Câu 2: Lưu lịch sử
                    st.session_state.chat_history.append({
                        "question":  question,
                        "answer":    answer,
                        "timestamp": datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                    })

                except Exception as e:
                    if "ollama" in str(e).lower() or "connection" in str(e).lower():
                        st.error("❌ Ollama chưa chạy! Mở terminal và chạy: `ollama serve`")
                    else:
                        st.error(f"❌ Lỗi: {str(e)}")
                        st.exception(e)
else:
    render_empty_state()

st.markdown("---")
st.caption("SmartDoc AI — Đại học Sài Gòn | OSSD Spring 2026 | LangChain + FAISS + Ollama")
