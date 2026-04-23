"""
app.py — Entry point của SmartDoc AI
Search Mode dùng key sạch từ sidebar, map sang build_retriever qua SEARCH_MODE_MAP.
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
    st.session_state.chat_history    = []
if "documents_store" not in st.session_state:
    st.session_state.documents_store = {}
if "all_chunks"      not in st.session_state:
    st.session_state.all_chunks      = []
if "uploader_key"    not in st.session_state:
    st.session_state.uploader_key    = 0

# ========================
# SIDEBAR
# ========================
config = render_sidebar()

chunk_strategy    = config["chunk_strategy"]
chunk_size        = config["chunk_size"]
chunk_overlap     = config["chunk_overlap"]
search_mode        = config["search_mode"]       # key sạch: "similarity"|"hybrid"|"mmr"|"graphrag_basic"|"graphrag_hybrid"
search_category    = config["search_category"]   # "rag" | "graphrag"
top_k             = config["top_k"]
use_rerank        = config["use_rerank"]
use_self_rag      = config["use_self_rag"]
use_conversational = config["use_conversational"]

# ========================
# MAPPING SEARCH MODE → build_retriever
# ========================
SEARCH_MODE_MAP = {
    "similarity":       "Similarity",
    "hybrid":           "Hybrid",
    "mmr":              "MMR",
    "graphrag_basic":   "GraphRAG",
    "graphrag_hybrid":  "GraphRAG_Hybrid",
}
real_search_mode = SEARCH_MODE_MAP.get(search_mode, "Similarity")

# ========================
# MAIN UI
# ========================
st.title("📄 SmartDoc AI")
st.markdown("**Hệ thống hỏi đáp thông minh** — RAG + Qwen2.5 | OSSD Spring 2026")
st.markdown("---")

# ── Upload tài liệu
st.subheader("📤 Upload tài liệu — PDF & DOCX ")
uploaded_files = st.file_uploader(
    "Chọn file PDF hoặc DOCX (có thể chọn nhiều file cùng lúc)",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    key=st.session_state.uploader_key
)

if uploaded_files:
    embedder = load_embedder()
    new_files = [f for f in uploaded_files if f.name not in st.session_state.documents_store]

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

                except Exception as e:
                    st.error(f"❌ Lỗi khi xử lý {uf.name}: {str(e)}")
    else:
        st.info("✅ Tất cả file đã được tải lên trước đó.")

# ── Q&A Section
if st.session_state.documents_store:
    st.markdown("---")

    # Lọc tài liệu
    all_doc_names = list(st.session_state.documents_store.keys())
    selected_docs = all_doc_names
    if len(all_doc_names) > 1:
        st.subheader("🔎 Lọc tài liệu")
        selected_docs = st.multiselect(
            "Tìm kiếm trong những tài liệu nào? (bỏ trống = tất cả)",
            all_doc_names,
            default=all_doc_names
        )

    st.subheader("💬 Đặt câu hỏi")

    render_chat_history(st.session_state.chat_history)

    question = st.text_input(
        "Nhập câu hỏi của bạn:",
        placeholder="Ví dụ: Tài liệu này nói về gì? Hoặc tóm tắt chương 2...",
        key="q_input"
    )

    if question:
        # Lấy chunks từ tài liệu được chọn
        selected_chunks = []
        for doc_name in (selected_docs if selected_docs else all_doc_names):
            selected_chunks.extend(st.session_state.documents_store[doc_name]["chunks"])

        if not selected_chunks:
            st.warning("⚠️ Không có nội dung nào để tìm kiếm.")
        else:
            with st.spinner("🤔 Đang tìm kiếm và sinh câu trả lời..."):
                try:
                    embedder = load_embedder()

                    # Với GraphRAG: lấy filename của file đầu tiên được chọn
                    # (GraphRAG build graph theo từng file riêng lẻ)
                    graph_filename = None
                    if search_category == "graphrag":
                        active_docs = selected_docs if selected_docs else all_doc_names
                        graph_filename = active_docs[0] if active_docs else None

                    retriever = build_retriever(
                        selected_chunks,
                        embedder,
                        real_search_mode,
                        top_k,
                        filename=graph_filename
                    )

                    answer, citations, self_eval, rewritten_q, elapsed = get_answer(
                        question, retriever,
                        use_rerank, use_self_rag, use_conversational,
                        top_k, st.session_state.chat_history,
                        search_category=search_category,
                        search_mode=search_mode,
                    )

                    render_rewrite_notice(question, rewritten_q)
                    render_answer(answer, elapsed)
                    render_self_rag_badge(self_eval)
                    render_citations(citations)

                    # Lưu lịch sử
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
st.caption("SmartDoc AI — Đại học Sài Gòn | OSSD Spring 2026 | LangChain + FAISS + Neo4j + Ollama")