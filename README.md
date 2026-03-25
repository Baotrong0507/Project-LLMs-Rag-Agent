# 📄 SmartDoc AI — Intelligent Document Q&A System

> Hệ thống hỏi đáp thông minh từ tài liệu sử dụng RAG (Retrieval-Augmented Generation)  
> Môn học: Open Source Software Development | Học kỳ: Spring 2026  
> Trường Đại học Sài Gòn — Khoa Công nghệ Thông tin

---

## 📋 Mục lục

- [Giới thiệu](#giới-thiệu)
- [Tính năng](#tính-năng)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Hướng dẫn cho Developers](#hướng-dẫn-cho-developers)
- [Chạy Tests](#chạy-tests)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)

---

## 🎯 Giới thiệu

**SmartDoc AI** là hệ thống RAG (Retrieval-Augmented Generation) cho phép người dùng upload tài liệu PDF/DOCX và đặt câu hỏi bằng ngôn ngữ tự nhiên. Hệ thống hỗ trợ tiếng Việt và tiếng Anh, chạy hoàn toàn **local** không cần internet sau khi cài đặt.

---

## ✨ Tính năng

| # | Tính năng | Mô tả |
|---|-----------|-------|
| 1 | **DOCX Support** | Upload và xử lý cả PDF lẫn DOCX |
| 2 | **Chat History** | Lưu trữ lịch sử hội thoại trong session |
| 3 | **Clear History** | Xóa lịch sử chat và tài liệu đã upload |
| 4 | **Chunk Strategy** | 3 chiến lược chunking: Recursive, Token, Paragraph |
| 5 | **Citation Tracking** | Hiển thị nguồn gốc trang/đoạn văn được dùng |
| 6 | **Conversational RAG** | Hỗ trợ hội thoại nhiều lượt với context |
| 7 | **Hybrid Search** | Kết hợp Vector Search + BM25 |
| 8 | **Multi-document RAG** | Upload và lọc nhiều tài liệu cùng lúc |
| 9 | **Re-ranking** | Cross-Encoder re-ranking kết quả retrieval |
| 10 | **Self-RAG** | LLM tự đánh giá + Query rewriting |

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
| GPU | Không bắt buộc | NVIDIA (tăng tốc) |

> ⚠️ **Lưu ý WSL (Windows):** Model Qwen2.5:7b và các package Python (~8GB) nên cài trên ổ D để tránh đầy ổ C.

---

## 🚀 Cài đặt

### Bước 1 — Clone repository

```bash
git clone https://github.com/baotrong0507/Project-LLMs-Rag-Agent.git
cd Project-LLMs-Rag-Agent
```

### Bước 2 — Tạo Virtual Environment

```bash
# Linux / Mac
python3 -m venv venv
source venv/bin/activate

# Windows (CMD)
python -m venv venv
venv\Scripts\activate

# WSL (khuyến nghị đặt venv trên ổ D)
python3 -m venv /mnt/d/venvs/smartdoc-env
source /mnt/d/venvs/smartdoc-env/bin/activate
```

### Bước 3 — Cài đặt Python dependencies

```bash
pip install -r requirements.txt
```

> ⏳ Quá trình này mất 5–15 phút tùy tốc độ mạng (tổng ~3–5 GB).  
> Nếu dùng WSL, thêm `--cache-dir /mnt/d/pip-cache` để cache sang ổ D:
> ```bash
> pip install -r requirements.txt --cache-dir /mnt/d/pip-cache
> ```

### Bước 4 — Cài đặt Ollama

**Linux / WSL:**
```bash
# Cài zstd trước (bắt buộc trên Ubuntu)
sudo apt-get install zstd -y

# Cài Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows / Mac:**  
Tải installer tại: https://ollama.ai/download

### Bước 5 — Pull model Qwen2.5:7b

```bash
# Chạy Ollama server (mở terminal riêng)
ollama serve

# Mở terminal mới, pull model (~4.7 GB)
ollama pull qwen2.5:7b
```

> ⏳ Download model mất 10–30 phút tùy tốc độ mạng. **Không tắt terminal.**

### Bước 6 — Chạy ứng dụng

Mở **2 terminal**:

**Terminal 1** — Chạy Ollama:
```bash
ollama serve
```

**Terminal 2** — Chạy Streamlit:
```bash
# Kích hoạt venv
source venv/bin/activate          # Linux/Mac
source /mnt/d/venvs/smartdoc-env/bin/activate  # WSL

# Chạy app
cd Project-LLMs-Rag-Agent
streamlit run app.py
```

Mở trình duyệt tại: **http://localhost:8501**

---

## 📁 Cấu trúc thư mục

```
Project-LLMs-Rag-Agent/
│
├── app.py                        # Main application file
├── requirements.txt              # Python dependencies
├── requirements_simplified.txt   # Simplified dependencies
├── README.md                     # Project documentation
├── test_smartdoc.py              # Test cases
│
├── data/                         # Data directory
│   └── gutenberg.pdf             # Sample PDF
│
├── documentation/                # Project documentation
│   ├── project_report.tex        # LaTeX report
│   └── README.md                 # Documentation guide
│
└── venv/                         # Virtual environment
```

---

## 📖 Hướng dẫn sử dụng

### Người dùng cuối

1. **Upload tài liệu** — Kéo thả hoặc click chọn file PDF/DOCX
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

```python
# Trong app.py, sửa hàm load_embedder()
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    # Thay bằng model khác:
    # model_name="sentence-transformers/all-MiniLM-L6-v2"       # Nhẹ hơn
    # model_name="sentence-transformers/all-mpnet-base-v2"       # Tiếng Anh tốt hơn
    model_kwargs={'device': 'cpu'},   # Đổi 'cuda' nếu có GPU
    encode_kwargs={'normalize_embeddings': True}
)
```

### 7.2.2 — Adjust Chunk Parameters

```python
# Trong app.py, sửa hàm split_documents()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # Thử: 500, 1000, 1500, 2000
    chunk_overlap=100    # Thử: 50, 100, 200
)
```

> **Gợi ý:** chunk nhỏ (500) → tìm kiếm chính xác hơn; chunk lớn (2000) → ngữ cảnh rộng hơn.

### 7.2.3 — Thay đổi LLM Model

```bash
# Pull model khác
ollama pull llama3:8b
ollama pull mistral:7b
ollama pull gemma2:9b
```

```python
# Trong app.py, sửa hàm get_answer()
llm = Ollama(
    model="qwen2.5:7b",     # Thay bằng: "llama3:8b", "mistral:7b"
    temperature=0.7,         # 0.0 = chính xác, 1.0 = sáng tạo
    top_p=0.9,
    repeat_penalty=1.1
)
```

### 7.2.4 — Modify Retrieval Parameters

```python
# Trong app.py, sửa hàm build_retriever()

# Similarity search (mặc định)
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}         # Tăng k để lấy nhiều kết quả hơn
)

# MMR — kết quả đa dạng hơn
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.7         # 0 = đa dạng, 1 = giống nhau
    }
)
```

### 7.2.5 — Add Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("smartdoc.log"),  # Ghi ra file
        logging.StreamHandler()               # Hiện trên terminal
    ]
)
logger = logging.getLogger(__name__)

# Dùng trong code
logger.info(f"Processing {len(documents)} chunks")
logger.info(f"Query: {user_input}")
logger.info(f"Retrieved {len(relevant_docs)} documents")
logger.warning("Ollama connection failed, retrying...")
logger.error(f"Error: {str(e)}")
```

---

## 🧪 Chạy Tests

### Cài đặt test dependencies

```bash
pip install pytest pytest-cov pytest-mock
```

### Chạy tất cả tests

```bash
pytest test_smartdoc.py -v
```

### Chạy từng nhóm

```bash
# Unit Testing & TDD
pytest test_smartdoc.py -v -k "Unit or TDD"

# Integration Testing
pytest test_smartdoc.py -v -k "Integration"

# Mock Testing
pytest test_smartdoc.py -v -k "Mock"

# Automation & CI/CD
pytest test_smartdoc.py -v -k "Automation or CI or Regression"
```

### Chạy với coverage report

```bash
pytest test_smartdoc.py --cov=app --cov-report=term
```

### Chạy thủ công (không cần pytest)

```bash
python test_smartdoc.py
```

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

## 📊 Performance

| Metric | Giá trị |
|--------|---------|
| PDF Loading | 2–5 giây |
| Embedding Generation | 5–10 giây / 100 chunks |
| Query Processing | 1–3 giây |
| Answer Generation | 3–8 giây |
| Retrieval Accuracy | 85–90% |
| Answer Relevance | 80–85% |

---

## 📝 License

MIT License — Free to use for educational purposes.

---

<div align="center">
  <strong>SmartDoc AI</strong> · Đại học Sài Gòn · OSSD Spring 2026<br>
  Powered by LangChain · FAISS · Ollama · Streamlit
</div>
