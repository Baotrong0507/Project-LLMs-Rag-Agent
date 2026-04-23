"""
ui/sidebar.py
Toàn bộ sidebar: settings, file list, history, nút xóa.
Search Mode chia 2 cấp: chế độ chính (RAG / GraphRAG) → sub-mode tương ứng.
Trả về search_category ("rag"|"graphrag") và search_mode (key sạch).
"""

import streamlit as st


def render_sidebar() -> dict:
    """
    Render sidebar và trả về dict config người dùng chọn.
    """
    with st.sidebar:
        st.title("⚙️ Cài đặt")
        st.markdown("---")

        # ── Câu 4: Chunk Strategy
        st.markdown("### 📐 Chunk Strategy")
        chunk_strategy = st.selectbox(
            "Chiến lược:",
            ["Recursive (Mặc định)", "Token-based", "Paragraph-based"]
        )
        chunk_size    = st.slider("Chunk Size",    500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap",  50,  300,  100,  50)

        st.markdown("---")

        # ── Câu 7: Search Mode (2 cấp: chế độ chính → sub-mode)
        st.markdown("### 🔍 Search Mode")

        # Cấp 1: chọn chế độ chính
        search_category = st.radio(
            "Chế độ chính:",
            ["🗂️ RAG (Vector-based)", "🕸️ GraphRAG"],
            horizontal=True,
            key="search_category"
        )

        # Cấp 2: sub-mode tương ứng
        if search_category == "🗂️ RAG (Vector-based)":
            RAG_SUB_MODES = {
                "Similarity (Mặc định)": "similarity",
                "Hybrid – Vector + BM25": "hybrid",
                "MMR – Đa dạng kết quả":  "mmr",
            }
            rag_choice = st.radio(
                "Chiến lược RAG:",
                list(RAG_SUB_MODES.keys()),
                index=0,
                key="rag_sub_mode",
                help=(
                    "**Similarity**: tìm kiếm thuần vector cosine.\n\n"
                    "**Hybrid**: kết hợp vector + BM25 keyword search.\n\n"
                    "**MMR**: tối đa hoá đa dạng, giảm trùng lặp."
                )
            )
            search_mode         = RAG_SUB_MODES[rag_choice]
            search_category_key = "rag"

        else:  # GraphRAG
            GRAPH_SUB_MODES = {
                "GraphRAG Cơ bản":                        "graphrag_basic",
                "GraphRAG + Vector Hybrid (Khuyến nghị)": "graphrag_hybrid",
            }
            graph_choice = st.radio(
                "Chiến lược GraphRAG:",
                list(GRAPH_SUB_MODES.keys()),
                index=1,
                key="graphrag_sub_mode",
                help=(
                    "**GraphRAG Cơ bản**: duyệt đồ thị tri thức Neo4j thuần túy.\n\n"
                    "**GraphRAG + Vector Hybrid**: kết hợp đồ thị + vector FAISS "
                    "để tăng độ chính xác."
                )
            )
            search_mode         = GRAPH_SUB_MODES[graph_choice]
            search_category_key = "graphrag"

        top_k = st.slider("Top K kết quả", 1, 10, 3)

        st.markdown("---")

        # ── Advanced Features
        st.markdown("### 🚀 Tính năng nâng cao")
        use_rerank         = st.checkbox("Re-ranking Cross-Encoder",  value=False)
        use_self_rag       = st.checkbox("Self-RAG Evaluation",       value=False)
        use_conversational = st.checkbox("Conversational RAG",        value=True)

        st.markdown("---")

        # ── Model Info
        st.markdown("### 🤖 Model Info")
        st.info("**LLM:** Qwen2.5:7b (Ollama)")
        st.info("**Embedding:** multilingual-mpnet (768d)")
        st.info("**Vector DB:** FAISS")
        st.info("**Graph DB:** Neo4j")

        st.markdown("---")

        # ── Câu 8: Documents uploaded
        st.markdown("### 📁 Tài liệu")
        if st.session_state.documents_store:
            for fname, info in st.session_state.documents_store.items():
                st.markdown(
                    f'<span class="doc-tag">📄 {fname}</span>',
                    unsafe_allow_html=True
                )
                st.caption(f"{info['num_chunks']} chunks | {info['upload_time']}")
        else:
            st.caption("Chưa có tài liệu")

        st.markdown("---")

        # ── Câu 2: Chat history
        st.markdown("### 💬 Lịch sử")
        if st.session_state.chat_history:
            st.caption(f"{len(st.session_state.chat_history)} câu hỏi")
            for i, h in enumerate(st.session_state.chat_history[-5:]):
                with st.expander(f"Q{i+1}: {h['question'][:35]}..."):
                    st.write(f"**Q:** {h['question']}")
                    st.write(f"**A:** {h['answer'][:150]}...")
                    st.caption(h['timestamp'])
        else:
            st.caption("Chưa có lịch sử")

        st.markdown("---")

        # ── Câu 3: Clear buttons
        st.markdown("### 🗑️ Xóa dữ liệu")
        _render_clear_buttons()

    return {
        "chunk_strategy":     chunk_strategy,
        "chunk_size":         chunk_size,
        "chunk_overlap":      chunk_overlap,
        "search_category":    search_category_key,   # "rag" | "graphrag"
        "search_mode":        search_mode,           # key sạch, ví dụ "similarity", "hybrid", ...
        "top_k":              top_k,
        "use_rerank":         use_rerank,
        "use_self_rag":       use_self_rag,
        "use_conversational": use_conversational,
    }


def _render_clear_buttons():
    """Câu 3: Nút xóa với confirmation dialog"""
    if "confirm_delete_chat" not in st.session_state:
        st.session_state.confirm_delete_chat = False
    if "confirm_delete_docs" not in st.session_state:
        st.session_state.confirm_delete_docs = False

    # Xóa Chat
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.confirm_delete_chat = True
        st.rerun()

    if st.session_state.confirm_delete_chat:
        st.warning("**Bạn chắc chắn muốn xóa toàn bộ lịch sử chat?**")
        st.caption("Hành động này không thể hoàn tác.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Xác nhận", key="confirm_chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.confirm_delete_chat = False
                st.success("✅ Đã xóa lịch sử chat!")
                st.rerun()
        with c2:
            if st.button("❌ Hủy", key="cancel_chat", use_container_width=True):
                st.session_state.confirm_delete_chat = False
                st.rerun()

    # Xóa Documents
    if st.button("🗑️ Xóa tất cả tài liệu", use_container_width=True):
        st.session_state.confirm_delete_docs = True
        st.rerun()

    if st.session_state.confirm_delete_docs:
        st.error("**Xóa TẤT CẢ tài liệu và vector store?**")
        st.caption("Toàn bộ file đã upload sẽ bị xóa!")
        d1, d2 = st.columns(2)
        with d1:
            if st.button("✅ Xác nhận", key="confirm_docs", use_container_width=True):
                st.session_state.documents_store = {}
                st.session_state.all_chunks = []
                if "uploader_key" not in st.session_state:
                    st.session_state.uploader_key = 0
                st.session_state.uploader_key += 1
                st.session_state.confirm_delete_docs = False
                st.success("✅ Đã xóa tất cả tài liệu!")
                st.rerun()
        with d2:
            if st.button("❌ Hủy", key="cancel_docs", use_container_width=True):
                st.session_state.confirm_delete_docs = False
                st.rerun()