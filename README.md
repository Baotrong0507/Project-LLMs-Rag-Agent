# 📄 SmartDoc AI — Intelligent Document Q&A System

> Hệ thống hỏi đáp thông minh từ tài liệu sử dụng RAG (Retrieval-Augmented Generation)  
> Môn học: Open Source Software Development | Học kỳ: Spring 2026  
> Trường Đại học Sài Gòn — Khoa Công nghệ Thông tin

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Tính năng](#-tính-năng)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Hướng dẫn cho Developers](#️-hướng-dẫn-cho-developers)
- [Chạy Tests](#-chạy-tests)

---

## 🎯 Giới thiệu

**SmartDoc AI** là hệ thống RAG (Retrieval-Augmented Generation) cho phép người dùng upload tài liệu PDF/DOCX và đặt câu hỏi bằng ngôn ngữ tự nhiên. Hệ thống hỗ trợ tiếng Việt và tiếng Anh, chạy hoàn toàn **local** không cần internet sau khi cài đặt.

---

## ✨ Tính năng

| # | Tính năng | Mô tả | Module |
|---|-----------|-------|--------|
| 1 | **DOCX Support** | Upload và xử lý cả PDF lẫn DOCX | `src/document_loader.py` |
| 2 | **Chat History** | Lưu trữ lịch sử hội thoại trong session | `ui/sidebar.py` |
| 3 | **Clear History** | Xóa lịch sử chat và tài liệu với confirmation | `ui/sidebar.py` |
| 4 | **Chunk Strategy** | 3 chiến lược: Recursive, Token, Paragraph | `src/chunker.py` |
| 5 | **Citation Tracking** | Hiển thị nguồn gốc trang/đoạn văn được dùng | `ui/components.py` |
| 6 | **Conversational RAG** | Hỗ trợ hội thoại nhiều lượt với context | `src/rag_engine.py` |
| 7 | **Hybrid Search** | Kết hợp Vector Search + BM25 | `src/retriever.py` |
| 8 | **Multi-document** | Upload và lọc nhiều tài liệu cùng lúc | `src/document_loader.py` |
| 9 | **Re-ranking** | Cross-Encoder re-ranking kết quả retrieval | `src/retriever.py` |
| 10 | **Self-RAG** | LLM tự đánh giá + Query rewriting | `src/rag_engine.py` |

---

## 🔧 Công nghệ sử dụng

| Thành phần | Công nghệ | Phiên bản |
|------------|-----------|-----------|
| Frontend | Streamlit | 1.41.1 |
| LLM Framework | LangChain | 0.3.16 |
| LLM Model | Qwen2.5:7b (Ollama) | Latest |
| Embedding | multilingual-mpnet-base-v2 | 768-dim |
| Vector DB | FAISS | 1.9.0 |
| PDF Loader | PDFPlumber | Latest |
| DOCX Loader | Docx2txt | Latest |
| BM25 Search | rank-bm25 | Latest |
| Re-ranking | CrossEncoder (sentence-transformers) | Latest |
| Testing | pytest + pytest-mock | Latest |

---

## 💻 Yêu cầu hệ thống

### Phần mềm bắt buộc

| Phần mềm | Phiên bản | Link tải |
|----------|-----------|----------|
| Python | 3.8+ | https://python.org |
| Ollama | Latest | https://ollama.ai |
| Git | Latest | https://git-scm.com |

### Phần cứng tối thiểu

| Thành phần | Tối thiểu | Khuyến nghị |
|------------|-----------|-------------|
| RAM | 8 GB | 16 GB |
| Dung lượng ổ cứng | 10 GB | 20 GB |
| CPU | 4 cores | 8 cores |

> ⚠️ **Lưu ý WSL (Windows):** Model Qwen2.5:7b và các package Python (~8GB) nên cài trên ổ D để tránh đầy ổ C.
---

## 🚀 Cài đặt

### Bước 1 — Clone repository

```bash
git clone https://github.com/Baotrong0507/Project-LLMs-Rag-Agent.git
cd Project-LLMs-Rag-Agent
```

### Bước 2 — Tạo Virtual Environment

```bash
# Linux / Mac
python3 -m venv venv
source venv/bin/activate

# WSL (khuyến nghị đặt venv trên ổ D)
python3 -m venv /mnt/d/venvs/smartdoc-env
source /mnt/d/venvs/smartdoc-env/bin/activate
```

### Bước 3 — Cài đặt Python dependencies

```bash
pip install -r requirements.txt

# WSL: cache sang ổ D để tiết kiệm ổ C
pip install -r requirements.txt --cache-dir /mnt/d/pip-cache
```

> ⏳ Quá trình này mất 5–15 phút (~3–5 GB).

### Bước 4 — Cài đặt Ollama

```bash
# Ubuntu / WSL
sudo apt-get install zstd -y
curl -fsSL https://ollama.com/install.sh | sh
```

### Bước 5 — Pull model Qwen2.5:7b

```bash
# Terminal 1: Chạy Ollama server
ollama serve

# Terminal 2: Pull model (~4.7 GB)
ollama pull qwen2.5:7b
```

### Bước 6 — Chạy ứng dụng

```bash
# Kích hoạt venv
source /mnt/d/venvs/smartdoc-env/bin/activate

# Chạy app
streamlit run app.py
```

Mở trình duyệt tại: **http://localhost:8501**

---

## 📁 Cấu trúc thư mục

```
Project-LLMs-Rag-Agent/
│
├── app.py                        # Entry point — luồng chính của ứng dụng
├── requirements.txt              # Python dependencies đầy đủ
├── requirements_simplified.txt   # Dependencies tối giản
├── README.md                     # Tài liệu hướng dẫn
├── test_smartdoc.py              # Test cases (Unit, Integration, Mock, CI/CD)
├── smartdoc.log                  # Log file (tự động tạo khi chạy)
│
├── src/                          # Toàn bộ business logic
│   ├── __init__.py
│   ├── logger.py                 # Cấu hình logging (7.2.5)
│   ├── document_loader.py        # Load PDF & DOCX (Câu 1)
│   ├── chunker.py                # Chunk strategy: Recursive/Token/Paragraph (Câu 4)
│   ├── retriever.py              # Hybrid search + Re-ranking (Câu 7, 9)
│   └── rag_engine.py             # RAG pipeline, Self-RAG, Query Rewrite (Câu 6, 10)
│
├── ui/                           # Toàn bộ giao diện người dùng
│   ├── __init__.py
│   ├── styles.py                 # CSS — chỉ sửa file này khi đổi giao diện
│   ├── sidebar.py                # Sidebar: settings, history, clear (Câu 2, 3, 8)
│   └── components.py             # Chat bubbles, answer, citation (Câu 5)
│
├── data/                         # Thư mục chứa tài liệu mẫu
│   └── gutenberg.pdf             # Sample PDF
│
└── documentation/                # Tài liệu kỹ thuật
    ├── project_report.tex        # Báo cáo LaTeX
    └── README.md                 # Hướng dẫn build báo cáo
```

---

## 📖 Hướng dẫn sử dụng

### Người dùng cuối

1. **Upload tài liệu** — Kéo thả hoặc click chọn file PDF/DOCX (nhiều file cùng lúc)
2. **Chờ xử lý** — Hệ thống tự động chunk, embed và index
3. **Đặt câu hỏi** — Nhập câu hỏi bằng tiếng Việt hoặc tiếng Anh
4. **Xem kết quả** — Câu trả lời + nguồn tham khảo hiển thị bên dưới


### Tips để câu trả lời chính xác hơn

- Đặt câu hỏi **cụ thể**, tránh quá chung chung
- Dùng **từ khóa** có trong tài liệu
- Chia câu hỏi phức tạp thành **nhiều câu nhỏ**



### Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|-----|-------------|------------|
| `Cannot connect to Ollama` | Ollama chưa chạy | Mở terminal, chạy `ollama serve` |
| Upload fail | File không phải PDF/DOCX | Kiểm tra định dạng file |
| Processing lâu | File quá lớn | Giới hạn file < 50MB |
| Không có response | Model chưa pull | Chạy `ollama pull qwen2.5:7b` |

---

## 🛠️ Hướng dẫn cho Developers

### 7.2.1 — Customize Embedding Model
**Sửa file:** `src/retriever.py`
```python
# Dòng 18 — đổi model tại đây
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# Ví dụ thay bằng:
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # Nhẹ hơn
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"   # Tiếng Anh tốt hơn
```

### 7.2.2 — Adjust Chunk Parameters
**Sửa file:** `src/chunker.py` hoặc kéo slider trong sidebar
```python
# Thử các giá trị chunk_size: 500, 1000, 1500, 2000
# Thử các giá trị chunk_overlap: 50, 100, 200
```
> **Gợi ý:** chunk nhỏ (500) → tìm kiếm chính xác hơn; chunk lớn (2000) → ngữ cảnh rộng hơn.
### 7.2.3 — Thay đổi LLM Model
**Sửa file:** `src/rag_engine.py`
```python
# Dòng 14 — đổi model tại đây
LLM_MODEL = "qwen2.5:7b"
# Sau khi: ollama pull llama3:8b
# Đổi thành: LLM_MODEL = "llama3:8b"
```

### 7.2.4 — Modify Retrieval Parameters
**Sửa file:** `src/retriever.py`
```python
# Trong hàm build_retriever() — sửa search_kwargs
search_kwargs={"k": top_k}           # Tăng k để lấy nhiều kết quả
search_kwargs={"lambda_mult": 0.7}   # MMR: 0=đa dạng, 1=giống nhau
weights=[0.4, 0.6]                   # Hybrid: BM25:Vector ratio
```

### 7.2.5 — Logging
**Sửa file:** `src/logger.py`
```python
# Đổi log level
level=logging.DEBUG    # Xem chi tiết hơn
level=logging.WARNING  # Chỉ xem cảnh báo
```
Log được ghi tự động vào file `smartdoc.log` và hiện trên terminal.

### Đổi giao diện
**Chỉ sửa file:** `ui/styles.py` — không cần đụng vào logic.

---

## 🧪 Chạy Tests
### Cài đặt test dependencies
```bash
pip install pytest pytest-cov pytest-mock

# Chạy tất cả
pytest test_smartdoc.py -v

# Chạy từng nhóm
pytest test_smartdoc.py -v -k "Unit or TDD"
pytest test_smartdoc.py -v -k "Integration"
pytest test_smartdoc.py -v -k "Mock"
pytest test_smartdoc.py -v -k "Automation or CI or Regression"

# Coverage report
pytest test_smartdoc.py --cov=src --cov-report=term

# Chạy thủ công (không cần pytest)
python test_smartdoc.py
```

---

## 📊 Performance

| Metric | Giá trị |
|--------|---------|
| PDF Loading | 2–5 giây |
| Embedding Generation | 5–10 giây / 100 chunks |
| Query Processing | 1–3 giây |
| Answer Generation | 3–8 giây |
| Retrieval Accuracy | 85–90% |

---

## 📝 License

MIT License — Free to use for educational purposes

---
<div align="center">
  <strong>SmartDoc AI</strong> · Đại học Sài Gòn · OSSD Spring 2026<br>
  Powered by LangChain · FAISS · Ollama · Streamlit
</div>
