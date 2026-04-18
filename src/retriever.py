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
    Tạo retriever theo search mode được chọn từ sidebar.
    Hỗ trợ cả hai nhóm: Vector-based và GraphRAG-based.
    """

    # ========================
    # Mapping search_mode từ sidebar (có emoji và khoảng trắng)
    # ========================
    mode = search_mode.strip()

    if "GraphRAG + Vector Hybrid" in mode:
        return _build_graph_hybrid_retriever(chunks, embedder, filename, top_k)
    
    elif "GraphRAG Cơ bản" in mode:
        return _build_graph_retriever(chunks, filename, top_k)

    # ========================
    # Vector-based RAG (FAISS)
    # ========================
    vector_store = FAISS.from_documents(chunks, embedder)

    if "Hybrid (Vector + BM25)" in mode:
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = top_k
        
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]          # BM25: 40% - Vector: 60%
        )

    elif "MMR (Đa dạng)" in mode:
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k":           top_k,
                "fetch_k":     top_k * 3,
                "lambda_mult": 0.7      # Càng nhỏ càng đa dạng
            }
        )

    else:  # Similarity (Mặc định) hoặc các mode khác
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

def _build_graph_hybrid_retriever(chunks: list, embedder, filename: str = None, top_k: int = 3):
    """GraphRAG + Vector Hybrid - Phiên bản ổn định"""
    from src.graph_rag import graph_retriever_for_file

    graph_retriever = _build_graph_retriever(chunks, filename, top_k)

    vector_store = FAISS.from_documents(chunks, embedder)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": top_k * 3})

    class HybridGraphRetriever:
        def __init__(self, graph_ret, vector_ret, top_k):
            self.graph_ret = graph_ret
            self.vector_ret = vector_ret
            self.top_k = top_k

        def invoke(self, query):
            return self.get_relevant_documents(query)

        def get_relevant_documents(self, query):
            try:
                graph_docs = self.graph_ret.get_relevant_documents(query)
            except:
                graph_docs = []

            # Nếu Graph trả về quá ít hoặc lỗi → lấy từ Vector
            if len(graph_docs) < self.top_k // 2:
                try:
                    vector_docs = self.vector_ret.get_relevant_documents(query)
                    # Kết hợp, ưu tiên graph_docs
                    combined = graph_docs + [d for d in vector_docs if d not in graph_docs]
                    return combined[:self.top_k * 2]
                except:
                    return graph_docs[:self.top_k]
            
            return graph_docs[:self.top_k]

    return HybridGraphRetriever(graph_retriever, vector_retriever, top_k)