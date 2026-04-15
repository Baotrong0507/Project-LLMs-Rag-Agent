"""
src/retriever.py
Câu 7: Hybrid search (Vector + BM25)
Câu 9: Re-ranking với Cross-Encoder
GraphRAG: Neo4j Knowledge Graph retrieval
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

# ========================
# CẤU HÌNH
# ========================

# 7.2.1: Đổi embedding model tại đây
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
RERANK_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Search modes có sẵn (bao gồm GraphRAG)
SEARCH_MODES = [
    "Similarity (Mặc định)",
    "Hybrid (Vector + BM25)",
    "MMR (Đa dạng)",
    "GraphRAG (Neo4j)"
]


# ========================
# LOAD MODELS (cached)
# ========================

@st.cache_resource(show_spinner=False)
def load_embedder():
    """Load embedding model — chỉ load 1 lần nhờ cache"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
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


# ========================
# CÂU 7: BUILD RETRIEVER
# ========================

def build_retriever(chunks: list, embedder, search_mode: str, top_k: int,
                    filename: str = None):
    """
    Tạo retriever theo search mode được chọn:
    - Similarity:        Vector search thuần (FAISS)
    - Hybrid:            BM25 + Vector ensemble
    - MMR:               Maximum Marginal Relevance
    - GraphRAG (Neo4j):  Knowledge Graph traversal

    7.2.4: Sửa search_kwargs bên dưới để thay đổi retrieval params
    """

    # GraphRAG — dùng Neo4j thay vì FAISS
    if search_mode == "GraphRAG (Neo4j)":
        if not filename:
            logger.warning("GraphRAG needs filename but got None")
        return _build_graph_retriever(chunks, filename, top_k)

    # Các mode còn lại dùng FAISS
    vector_store = FAISS.from_documents(chunks, embedder)

    if search_mode == "Hybrid (Vector + BM25)":
        vector_retriever = vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )
        bm25_retriever   = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = top_k
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]          # BM25:40% Vector:60%
        )

    elif search_mode == "MMR (Đa dạng)":
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k":           top_k,
                "fetch_k":     top_k * 3,
                "lambda_mult": 0.7      # 0=đa dạng, 1=giống nhau
            }
        )

    else:  # Similarity (mặc định)
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )

    return retriever


def _build_graph_retriever(chunks: list, filename: str = None, top_k: int = 3):
    from src.graph_rag import build_graph_from_chunks, graph_retriever_for_file

    if filename:
        with st.spinner("🔗 Đang xây dựng Knowledge Graph..."):
            build_graph_from_chunks(chunks, filename)

    class GraphRetriever:
        def __init__(self, filename=None, top_k=3):
            self.filename = filename
            self.top_k = top_k

        def invoke(self, query):
            return self.get_relevant_documents(query)

        def get_relevant_documents(self, query):
            return graph_retriever_for_file(query, self.filename, self.top_k)

    return GraphRetriever(filename, top_k)

# ========================
# CÂU 9: RE-RANKING
# ========================

def rerank_documents(query: str, docs: list, cross_encoder, top_k: int) -> list:
    """
    Re-rank documents dùng Cross-Encoder.
    Score cao hơn = liên quan hơn với query.
    """
    if cross_encoder is None or len(docs) == 0:
        return docs[:top_k]

    pairs  = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    logger.info(f"Re-ranked {len(docs)} docs → top {top_k}")
    return [doc for _, doc in scored[:top_k]]