"""
src/retriever.py
Câu 7: Hybrid search (Vector + BM25)
Câu 9: Re-ranking với Cross-Encoder
"""
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from src.logger import logger

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

SEARCH_MODES = ["Similarity (Mặc định)", "Hybrid (Vector + BM25)", "MMR (Đa dạng)"]

# 7.2.1: Muốn đổi embedding model? Chỉ sửa MODEL_NAME ở đây
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
RERANK_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@st.cache_resource(show_spinner=False)
def load_embedder():
    """Load embedding model (cached, chỉ load 1 lần)"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},   # 7.2.1: đổi 'cuda' nếu có GPU
        encode_kwargs={'normalize_embeddings': True}
    )


@st.cache_resource(show_spinner=False)
def load_cross_encoder():
    """Load cross-encoder cho re-ranking (Câu 9)"""
    if not CROSS_ENCODER_AVAILABLE:
        return None
    try:
        return CrossEncoder(RERANK_MODEL)
    except Exception:
        return None


def build_retriever(chunks: list, embedder, search_mode: str, top_k: int):
    """
    Câu 7: Tạo retriever theo search mode.
    - Similarity: vector search thuần
    - Hybrid: BM25 + vector (weighted ensemble)
    - MMR: Maximum Marginal Relevance (đa dạng kết quả)

    7.2.4: Muốn sửa retrieval params? Sửa search_kwargs bên dưới.
    """
    vector_store = FAISS.from_documents(chunks, embedder)

    if search_mode == "Hybrid (Vector + BM25)":
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        bm25_retriever   = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = top_k
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]      # BM25:40% Vector:60%
        )
    elif search_mode == "MMR (Đa dạng)":
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k":           top_k,
                "fetch_k":     top_k * 3,
                "lambda_mult": 0.7      # 0=đa dạng tối đa, 1=giống nhau
            }
        )
    else:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )

    return retriever


def rerank_documents(query: str, docs: list, cross_encoder, top_k: int) -> list:
    """
    Câu 9: Re-rank documents dùng Cross-Encoder.
    Score cao hơn = liên quan hơn với query.
    """
    if cross_encoder is None or len(docs) == 0:
        return docs[:top_k]

    pairs  = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    logger.info(f"Re-ranked {len(docs)} docs, top_k={top_k}")
    return [doc for _, doc in scored[:top_k]]
