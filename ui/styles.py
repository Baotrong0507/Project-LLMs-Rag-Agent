"""
ui/styles.py
Toàn bộ CSS của ứng dụng.
Muốn đổi giao diện? Chỉ sửa file này, không cần đụng vào logic.
"""
import streamlit as st


def load_styles():
    """Inject CSS vào Streamlit app"""
    st.markdown("""
<style>
    .main { background-color: #F8F9FA; }

    /* Answer box */
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

    /* Chat bubbles */
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

    /* Citation */
    .citation-box {
        background-color: #FFF3CD;
        color: #212529;
        border-left: 3px solid #FFC107;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 13px;
        margin: 4px 0;
    }

    /* Buttons */
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover { background-color: #0056b3; }

    /* Doc tag */
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