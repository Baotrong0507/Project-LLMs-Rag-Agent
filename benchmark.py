
import time
import sys
sys.path.insert(0, '.')

from src.document_loader import load_document
from src.chunker import split_documents
from src.retriever import load_embedder, build_retriever

# ========================
# CÁC CÂU HỎI TEST
# ========================
TEST_QUESTIONS = [
    "RAG là gì?",
    "Hệ thống sử dụng công nghệ gì?",
    "FAISS hoạt động như thế nào?",
    "Các tính năng của hệ thống?",
    "LangChain framework là gì?",
]

print("Đang tải tài liệu và chunks...")
with open("data/gutenberg.pdf", "rb") as f:
    file_bytes = f.read()

docs = load_document(file_bytes, "gutenberg.pdf")
chunks = split_documents(docs, "Recursive (Mặc định)", 1000, 100)

embedder = load_embedder()

print(f"Đã load {len(chunks)} chunks từ gutenberg.pdf\n")

# ========================
# DANH SÁCH CÁC MODE CẦN TEST (ĐÃ CẬP NHẬT)
# ========================
MODES = [
    "   • Similarity (Mặc định)",
    "   • Hybrid (Vector + BM25)",
    "   • MMR (Đa dạng)",
    "   • GraphRAG Cơ bản",
    "   • GraphRAG + Vector Hybrid (Khuyến nghị)",
]

print(f"{'Chế độ':<40} {'TB (s)':<10} {'Min (s)':<10} {'Max (s)':<10}")
print("-" * 75)

for mode in MODES:
    times = []
    print(f"Đang test: {mode.strip()}")
    
    for q in TEST_QUESTIONS:
        t0 = time.time()
        
        try:
            # Truyền filename cho GraphRAG
            retriever = build_retriever(
                chunks, 
                embedder, 
                mode, 
                top_k=3, 
                filename="gutenberg.pdf"
            )
            
            # Gọi retriever
            if hasattr(retriever, 'invoke'):
                docs_ret = retriever.invoke(q)
            else:
                docs_ret = retriever.get_relevant_documents(q)
                
            elapsed = time.time() - t0
            times.append(elapsed)
            
        except Exception as e:
            elapsed = time.time() - t0
            times.append(elapsed)
            print(f"   Lỗi với câu hỏi: {q[:50]}... → {type(e).__name__}")
    
    avg = sum(times) / len(times)
    print(f"{mode:<40} {avg:<10.2f} {min(times):<10.2f} {max(times):<10.2f}")

print("\n✅ Benchmark hoàn tất!")
