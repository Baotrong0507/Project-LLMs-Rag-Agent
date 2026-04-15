"""
ui/sidebar.py
Toàn bộ sidebar: settings, file list, history, nút xóa.
Câu 2, 3, 4, 6, 7, 8, 9, 10
"""
import streamlit as st


def render_sidebar() -> dict:
    """
    Render sidebar và trả về dict config người dùng chọn.
    Tách riêng để main app chỉ cần: config = render_sidebar()
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

        # ── Câu 7: Search Mode
        st.markdown("### 🔍 Search Mode")
        search_mode = st.selectbox(
            "Chế độ tìm kiếm:",
            ["Similarity (Mặc định)", "Hybrid (Vector + BM25)", "MMR (Đa dạng)", "GraphRAG (Neo4j)"]
        )
        top_k = st.slider("Top K kết quả", 1, 10, 3)

        st.markdown("---")

        # ── Advanced Features
        st.markdown("### 🚀 Tính năng nâng cao")
        use_rerank         = st.checkbox("Re-ranking Cross-Encoder",  value=False)
        use_self_rag       = st.checkbox("Self-RAG Evaluation)",      value=False)
        use_conversational = st.checkbox("Conversational RAG",        value=True)

        st.markdown("---")

        # ── Model Info
        st.markdown("### 🤖 Model Info")
        st.info("**LLM:** Qwen2.5:7b (Ollama)")
        st.info("**Embedding:** multilingual-mpnet (768d)")
        st.info("**Vector DB:** FAISS")

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

        # ── Câu 3: Clear buttons với confirmation dialog
        st.markdown("### 🗑️ Xóa dữ liệu")
        _render_clear_buttons()

    return {
        "chunk_strategy":    chunk_strategy,
        "chunk_size":        chunk_size,
        "chunk_overlap":     chunk_overlap,
        "search_mode":       search_mode,
        "top_k":             top_k,
        "use_rerank":        use_rerank,
        "use_self_rag":      use_self_rag,
        "use_conversational": use_conversational,
    }


def _render_clear_buttons():
    """Câu 3: Nút xóa với confirmation dialog"""

    # Khởi tạo session state
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
                st.session_state.chat_history        = []
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
                st.session_state.documents_store     = {}
                st.session_state.all_chunks          = []
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