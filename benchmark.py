import time
import sys
sys.path.insert(0, '.')

from src.document_loader import load_document
from src.chunker         import split_documents
from src.retriever       import load_embedder, build_retriever

# ========================
# CÁC CÂU HỎI TEST
# ========================
TEST_QUESTIONS = [
    "RAG là gì?",
    "Hệ thống sử dụng công nghệ gì?",
    "FAISS hoạt động như thế nào?",
    "Các tính năng của hệ thống?",
    "LangChain framework là gì?",
    "Tóm tắt nội dung của tài liệu",
]

print("Đang tải tài liệu và chunks...")
with open("data/gutenberg.pdf", "rb") as f:
    file_bytes = f.read()

docs   = load_document(file_bytes, "gutenberg.pdf")
chunks = split_documents(docs, "Recursive (Mặc định)", 1000, 100)
embedder = load_embedder()
print(f"Đã load {len(chunks)} chunks từ gutenberg.pdf\n")

# ========================
# DANH SÁCH MODE — dùng clean key khớp với retriever.py đã fix
# ========================
MODES = [
    ("Similarity (Mặc định)",          "similarity"),
    ("Hybrid (Vector + BM25)",         "hybrid"),
    ("MMR (Đa dạng)",                  "mmr"),
    ("GraphRAG Cơ bản",                "graphrag_basic"),
    ("GraphRAG + Vector Hybrid",       "graphrag_hybrid"),
]

print(f"{'Chế độ':<35} {'TB (s)':<10} {'Min (s)':<10} {'Max (s)':<10} {'Lỗi'}")
print("-" * 80)

for label, mode_key in MODES:
    times  = []
    errors = 0
    print(f"Đang test: {label}")
 
    for q in TEST_QUESTIONS:
        t0 = time.time()
        try:
            retriever = build_retriever(
                chunks,
                embedder,
                mode_key,           # ← clean key
                top_k=3,
                filename="gutenberg.pdf"
            )
            if hasattr(retriever, "invoke"):
                retriever.invoke(q)
            else:
                retriever.get_relevant_documents(q)
        except Exception as e:
            errors += 1
            print(f"   ⚠ Lỗi '{q[:40]}': {type(e).__name__}: {e}")
        finally:
            times.append(time.time() - t0)

    avg = sum(times) / len(times)
    print(f"{label:<35} {avg:<10.2f} {min(times):<10.2f} {max(times):<10.2f} {errors}/{len(TEST_QUESTIONS)}")

print("\n✅ Benchmark hoàn tất!")
