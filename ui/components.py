"""
ui/components.py
Các UI component tái sử dụng:
- render_chat_history()   — Câu 2
- render_answer()         — hiển thị câu trả lời
- render_citations()      — Câu 5
- render_self_rag_badge() — Câu 10
- render_empty_state()    — khi chưa upload file
"""
import streamlit as st


def render_chat_history(chat_history: list):
    """Câu 2: Hiển thị lịch sử hội thoại dạng chat bubble"""
    if not chat_history:
        return
    st.markdown("#### 📜 Lịch sử hội thoại")
    for h in chat_history:
        st.markdown(
            f'<div class="chat-user">🧑 {h["question"]}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="chat-bot">🤖 {h["answer"]}</div>',
            unsafe_allow_html=True
        )
    st.markdown("---")


def render_answer(answer: str, elapsed: float):
    """Hiển thị câu trả lời với thời gian xử lý"""
    st.subheader("💡 Câu trả lời")
    st.markdown(
        f'<div class="answer-box">{answer}</div>',
        unsafe_allow_html=True
    )
    st.caption(f"⏱️ Thời gian xử lý: {elapsed:.1f} giây")


def render_citations(citations: list):
    """Câu 5: Hiển thị nguồn tham khảo"""
    with st.expander(f"📚 Nguồn tham khảo — {len(citations)} đoạn (Câu 5)"):
        for c in citations:
            st.markdown(f"""
            <div class="citation-box">
                <strong>[{c['index']}] 📄 {c['source']} — Trang {c['page']}</strong><br>
                <em>"{c['content']}..."</em>
            </div>
            """, unsafe_allow_html=True)


def render_self_rag_badge(self_eval: dict):
    """Câu 10: Hiển thị kết quả Self-RAG evaluation"""
    if not self_eval:
        return
    score = self_eval.get("score", 0)
    icon  = "🟢" if score >= 7 else "🟡" if score >= 5 else "🔴"
    reason = self_eval.get("reason", "")
    st.info(f"{icon} **Self-RAG (Câu 10):** Điểm {score}/10 — {reason}")


def render_empty_state():
    """Hiển thị khi chưa có tài liệu nào được upload"""
    st.info("👆 Vui lòng upload ít nhất một file PDF hoặc DOCX để bắt đầu.")


def render_rewrite_notice(original: str, rewritten: str):
    """Câu 10: Hiển thị thông báo khi query được viết lại"""
    if rewritten != original:
        st.caption(f"🔄 Câu hỏi được viết lại (Câu 10): *{rewritten}*")