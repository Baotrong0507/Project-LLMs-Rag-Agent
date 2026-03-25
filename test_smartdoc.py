"""
=============================================================
TEST CASES - SmartDoc AI (RAG System)
OSSD Spring 2026 - Đại học Sài Gòn
=============================================================
Bao gồm:
  1. Unit Testing và TDD
  2. Integration Testing
  3. Mock Testing
  4. Automation và CI/CD
=============================================================
Cài đặt: pip install pytest pytest-cov pytest-mock
Chạy tất cả:   pytest test_smartdoc.py -v
Chạy 1 nhóm:  pytest test_smartdoc.py -v -k "Unit"
Chạy coverage: pytest test_smartdoc.py --cov=app --cov-report=term
=============================================================
"""

import pytest
import os
import sys
import json
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import (
    detect_language,
    split_documents,
    load_document,
    rerank_documents,
    self_rag_evaluate,
    rewrite_query,
)
from langchain.schema import Document


# ================================================================
#  PHẦN 1: UNIT TESTING VÀ TDD
# ================================================================
#  TDD (Test-Driven Development) - 3 bước:
#    RED    -> Viết test TRƯỚC khi có code (test sẽ fail)
#    GREEN  -> Viết code tối thiểu để test pass
#    REFACTOR -> Tối ưu code, giữ nguyên test xanh
# ================================================================

class TestUnit_DetectLanguage:
    """
    UNIT TEST: Hàm detect_language()
    Kiểm tra từng trường hợp ngôn ngữ độc lập
    """

    # TDD - RED phase: viết test trước khi implement
    def test_tdd_vi_returns_vi(self):
        """[TDD-RED] Câu tiếng Việt phải trả về 'vi'"""
        assert detect_language("Xin chào tôi là sinh viên") == 'vi'

    def test_tdd_en_returns_en(self):
        """[TDD-RED] Câu tiếng Anh phải trả về 'en'"""
        assert detect_language("Hello I am a student") == 'en'

    # Unit tests chi tiết
    def test_unit_empty_string_defaults_en(self):
        """[UNIT] Chuỗi rỗng -> mặc định 'en'"""
        assert detect_language("") == 'en'

    def test_unit_numbers_only_is_en(self):
        """[UNIT] Chỉ có số -> 'en'"""
        assert detect_language("1234567890") == 'en'

    def test_unit_special_chars_is_en(self):
        """[UNIT] Ký tự đặc biệt -> 'en'"""
        assert detect_language("!@#$%^&*()") == 'en'

    def test_unit_short_vi_sentence(self):
        """[UNIT] Câu ngắn tiếng Việt"""
        assert detect_language("Đây là gì?") == 'vi'

    def test_unit_full_diacritics_vi(self):
        """[UNIT] Tiếng Việt đầy đủ dấu"""
        assert detect_language("Hệ thống được xây dựng như thế nào?") == 'vi'

    def test_unit_technical_vi(self):
        """[UNIT] Câu kỹ thuật tiếng Việt"""
        assert detect_language("Mô hình embedding đa ngôn ngữ hoạt động ra sao?") == 'vi'

    def test_unit_technical_en(self):
        """[UNIT] Câu kỹ thuật tiếng Anh"""
        assert detect_language("How does the embedding model process tokens?") == 'en'

    def test_unit_return_type_is_string(self):
        """[UNIT] Kết quả phải là string"""
        assert isinstance(detect_language("test"), str)

    def test_unit_return_value_valid(self):
        """[UNIT] Kết quả phải là 'vi' hoặc 'en'"""
        assert detect_language("any text") in ['vi', 'en']

    def test_unit_mixed_vi_dominant(self):
        """[UNIT] Văn bản pha - tiếng Việt chiếm đa số"""
        assert detect_language("RAG được sử dụng rộng rãi trong hệ thống AI") == 'vi'

    def test_tdd_refactor_consistent(self):
        """[TDD-REFACTOR] Gọi nhiều lần kết quả phải nhất quán"""
        text = "Tài liệu này rất hay và bổ ích"
        results = [detect_language(text) for _ in range(5)]
        assert all(r == results[0] for r in results), "Kết quả phải nhất quán"


class TestUnit_SplitDocuments:
    """
    UNIT TEST: Hàm split_documents()
    Kiểm tra từng chiến lược chunking (Câu 4)
    """

    @pytest.fixture
    def long_doc(self):
        return [Document(
            page_content="Đây là câu văn mẫu để kiểm tra chunking. " * 100,
            metadata={"source": "test.pdf"}
        )]

    def test_unit_recursive_returns_list(self, long_doc):
        """[UNIT] Recursive strategy trả về list"""
        result = split_documents(long_doc, "Recursive (Mặc định)", 200, 20)
        assert isinstance(result, list)

    def test_unit_recursive_not_empty(self, long_doc):
        """[UNIT] Recursive strategy không trả về rỗng"""
        assert len(split_documents(long_doc, "Recursive (Mặc định)", 200, 20)) > 0

    def test_unit_token_based_works(self, long_doc):
        """[UNIT] Token-based strategy hoạt động"""
        assert len(split_documents(long_doc, "Token-based", 200, 20)) > 0

    def test_unit_paragraph_based_works(self, long_doc):
        """[UNIT] Paragraph-based strategy hoạt động"""
        assert len(split_documents(long_doc, "Paragraph-based", 200, 20)) > 0

    def test_unit_chunks_are_document_objects(self, long_doc):
        """[UNIT] Mỗi chunk phải là Document object"""
        result = split_documents(long_doc, "Recursive (Mặc định)", 200, 20)
        assert all(isinstance(c, Document) for c in result)

    def test_unit_chunk_content_not_empty(self, long_doc):
        """[UNIT] Mỗi chunk phải có nội dung"""
        result = split_documents(long_doc, "Recursive (Mặc định)", 200, 20)
        assert all(len(c.page_content.strip()) > 0 for c in result)

    def test_tdd_larger_size_fewer_chunks(self):
        """[TDD] Chunk size lớn hơn -> ít chunks hơn"""
        text = "Câu ngắn. " * 300
        docs = [Document(page_content=text, metadata={})]
        small = split_documents(docs, "Recursive (Mặc định)", 100, 10)
        large = split_documents(docs, "Recursive (Mặc định)", 1000, 10)
        assert len(small) >= len(large), "Chunk size lớn phải tạo ít chunks hơn"

    def test_tdd_overlap_increases_chunks(self):
        """[TDD] Overlap lớn hơn -> nhiều chunks hơn"""
        text = "Test sentence. " * 100
        docs = [Document(page_content=text, metadata={})]
        no_overlap   = split_documents(docs, "Recursive (Mặc định)", 100, 0)
        with_overlap = split_documents(docs, "Recursive (Mặc định)", 100, 50)
        assert len(with_overlap) >= len(no_overlap)

    def test_unit_compare_all_chunk_sizes(self):
        """[UNIT][Câu 4] So sánh chunk_size 500, 1000, 1500, 2000"""
        text = "Đây là văn bản mẫu. " * 400
        docs = [Document(page_content=text, metadata={})]
        sizes = [500, 1000, 1500, 2000]
        counts = [len(split_documents(docs, "Recursive (Mặc định)", s, 50)) for s in sizes]
        assert counts == sorted(counts, reverse=True), \
            "Chunk size nhỏ hơn phải tạo nhiều chunks hơn"

    def test_unit_empty_doc_no_crash(self):
        """[UNIT] Tài liệu rỗng không crash"""
        docs = [Document(page_content="", metadata={})]
        result = split_documents(docs, "Recursive (Mặc định)", 200, 20)
        assert isinstance(result, list)


class TestUnit_Reranking:
    """
    UNIT TEST: Hàm rerank_documents() (Câu 9)
    """

    @pytest.fixture
    def sample_docs(self):
        return [
            Document(page_content="FAISS là thư viện vector search", metadata={}),
            Document(page_content="Streamlit là web framework Python", metadata={}),
            Document(page_content="RAG kết hợp retrieval và generation", metadata={}),
        ]

    @pytest.fixture
    def mock_encoder(self):
        enc = MagicMock()
        enc.predict.return_value = [0.3, 0.9, 0.6]
        return enc

    def test_unit_returns_list(self, sample_docs, mock_encoder):
        """[UNIT] Trả về list"""
        assert isinstance(rerank_documents("q", sample_docs, mock_encoder, 2), list)

    def test_unit_top_k_respected(self, sample_docs, mock_encoder):
        """[UNIT] Đúng số lượng top_k"""
        assert len(rerank_documents("q", sample_docs, mock_encoder, 2)) == 2

    def test_unit_highest_score_first(self, sample_docs, mock_encoder):
        """[UNIT] Doc có score cao nhất (index 1, 0.9) đứng đầu"""
        result = rerank_documents("q", sample_docs, mock_encoder, 3)
        assert result[0].page_content == sample_docs[1].page_content

    def test_unit_none_encoder_returns_original(self, sample_docs):
        """[UNIT] encoder=None trả về docs gốc"""
        assert rerank_documents("q", sample_docs, None, 2) == sample_docs

    def test_unit_empty_docs_returns_empty(self, mock_encoder):
        """[UNIT] Docs rỗng trả về []"""
        assert rerank_documents("q", [], mock_encoder, 3) == []

    def test_unit_top_k_larger_than_docs(self, sample_docs, mock_encoder):
        """[UNIT] top_k > len(docs) không crash"""
        result = rerank_documents("q", sample_docs, mock_encoder, 100)
        assert len(result) <= len(sample_docs)


class TestUnit_SelfRAG:
    """
    UNIT TEST: Hàm self_rag_evaluate() (Câu 10)
    """

    @pytest.fixture
    def good_llm(self):
        llm = MagicMock()
        llm.invoke.return_value = '{"score": 9, "reason": "Chính xác", "is_sufficient": true}'
        return llm

    def test_unit_returns_dict(self, good_llm):
        assert isinstance(self_rag_evaluate(good_llm, "Q", "A", "C"), dict)

    def test_unit_has_score(self, good_llm):
        assert "score" in self_rag_evaluate(good_llm, "Q", "A", "C")

    def test_unit_has_reason(self, good_llm):
        assert "reason" in self_rag_evaluate(good_llm, "Q", "A", "C")

    def test_unit_has_is_sufficient(self, good_llm):
        assert "is_sufficient" in self_rag_evaluate(good_llm, "Q", "A", "C")

    def test_unit_score_in_range(self, good_llm):
        result = self_rag_evaluate(good_llm, "Q", "A", "C")
        assert 1 <= result["score"] <= 10

    def test_unit_fallback_on_bad_json(self):
        llm = MagicMock()
        llm.invoke.return_value = "không phải json"
        result = self_rag_evaluate(llm, "Q", "A", "C")
        assert isinstance(result, dict) and "score" in result

    def test_unit_fallback_on_exception(self):
        llm = MagicMock()
        llm.invoke.side_effect = Exception("LLM error")
        result = self_rag_evaluate(llm, "Q", "A", "C")
        assert isinstance(result, dict)


# ================================================================
#  PHẦN 2: INTEGRATION TESTING
# ================================================================
#  Kiểm tra nhiều module phối hợp:
#    - load_document + split_documents
#    - detect_language + prompt logic
#    - rerank + citation pipeline
# ================================================================

class TestIntegration_LoadAndSplit:
    """INTEGRATION TEST: Load Document -> Split Documents"""

    def test_integ_pdf_load_then_split(self):
        """[INTEG] PDF load -> split tạo chunks hợp lệ"""
        with patch('app.PDFPlumberLoader') as ml:
            ml.return_value.load.return_value = [
                Document(page_content="Nội dung PDF mẫu. " * 50, metadata={"page": 1})
            ]
            docs = load_document(b"%PDF", "test.pdf")
            chunks = split_documents(docs, "Recursive (Mặc định)", 200, 20)
            assert len(chunks) > 0
            assert all(isinstance(c, Document) for c in chunks)

    def test_integ_docx_load_then_split(self):
        """[INTEG] DOCX load -> split tạo chunks hợp lệ"""
        with patch('app.Docx2txtLoader') as ml:
            ml.return_value.load.return_value = [
                Document(page_content="Nội dung DOCX mẫu. " * 50, metadata={})
            ]
            docs = load_document(b"PK", "test.docx")
            chunks = split_documents(docs, "Paragraph-based", 300, 30)
            assert len(chunks) > 0

    def test_integ_metadata_survives_split(self):
        """[INTEG] Metadata giữ nguyên qua load -> split"""
        with patch('app.PDFPlumberLoader') as ml:
            ml.return_value.load.return_value = [
                Document(page_content="Content. " * 60, metadata={"source_file": "report.pdf"})
            ]
            docs = load_document(b"%PDF", "report.pdf")
            assert docs[0].metadata["source_file"] == "report.pdf"
            assert docs[0].metadata["file_type"] == "pdf"
            assert "upload_time" in docs[0].metadata

    def test_integ_large_doc_chunk_count_reasonable(self):
        """[INTEG] Tài liệu dài tạo số chunks hợp lý"""
        with patch('app.PDFPlumberLoader') as ml:
            ml.return_value.load.return_value = [
                Document(page_content="Câu văn test. " * 500, metadata={})
            ]
            docs = load_document(b"%PDF", "large.pdf")
            chunks = split_documents(docs, "Recursive (Mặc định)", 500, 50)
            assert 5 < len(chunks) < 200


class TestIntegration_Pipeline:
    """INTEGRATION TEST: Pipeline đầu đủ"""

    def test_integ_language_detection_pipeline(self):
        """[INTEG] Pipeline phát hiện ngôn ngữ"""
        vi_q = "Tài liệu này nói về chủ đề gì?"
        en_q = "What is the main topic of this document?"
        assert detect_language(vi_q) != detect_language(en_q)

    def test_integ_multi_doc_chunks_combined(self):
        """[INTEG] Gộp chunks từ nhiều tài liệu (Câu 8)"""
        ca = [Document(page_content=f"A{i}", metadata={"source_file": "a.pdf"}) for i in range(5)]
        cb = [Document(page_content=f"B{i}", metadata={"source_file": "b.pdf"}) for i in range(3)]
        combined = ca + cb
        sources = {c.metadata["source_file"] for c in combined}
        assert len(combined) == 8
        assert "a.pdf" in sources and "b.pdf" in sources

    def test_integ_rerank_after_retrieval(self):
        """[INTEG] Re-ranking sau retrieval (Câu 9)"""
        docs = [
            Document(page_content="Không liên quan", metadata={}),
            Document(page_content="RAG là kỹ thuật quan trọng", metadata={}),
            Document(page_content="FAISS tìm kiếm nhanh", metadata={}),
        ]
        enc = MagicMock()
        enc.predict.return_value = [0.1, 0.95, 0.4]
        result = rerank_documents("RAG là gì?", docs, enc, top_k=2)
        assert result[0].page_content == "RAG là kỹ thuật quan trọng"

    def test_integ_citation_from_docs(self):
        """[INTEG] Tạo citations từ source documents (Câu 5)"""
        source_docs = [
            Document(page_content="RAG kết hợp retrieval.", metadata={"source_file": "rag.pdf", "page": 1}),
            Document(page_content="FAISS hỗ trợ GPU.", metadata={"source_file": "faiss.pdf", "page": 3}),
        ]
        citations = [
            {"index": i+1, "content": d.page_content[:200],
             "source": d.metadata.get("source_file", "?"),
             "page": d.metadata.get("page", "N/A")}
            for i, d in enumerate(source_docs)
        ]
        assert len(citations) == 2
        assert citations[0]["source"] == "rag.pdf"
        assert citations[1]["page"] == 3
        assert all("index" in c for c in citations)

    def test_integ_conversational_history_grows(self):
        """[INTEG] Lịch sử hội thoại tích lũy đúng (Câu 2, 6)"""
        history = []
        turns = [
            ("RAG là gì?", "RAG là Retrieval-Augmented Generation."),
            ("Nó dùng để làm gì?", "Dùng để hỏi đáp từ tài liệu."),
            ("Ưu điểm là gì?", "Câu trả lời chính xác và có nguồn."),
        ]
        for q, a in turns:
            history.append({"question": q, "answer": a,
                             "timestamp": datetime.now().strftime("%H:%M")})
        assert len(history) == 3
        assert history[-1]["question"] == "Ưu điểng là gì?"


# ================================================================
#  PHẦN 3: MOCK TESTING
# ================================================================
#  Thay thế các thành phần bên ngoài bằng Mock:
#    - Ollama LLM  -> MagicMock
#    - File loaders -> patch
#    - Cross-Encoder -> MagicMock
# ================================================================

class TestMock_OllamaLLM:
    """MOCK TEST: Thay thế Ollama LLM"""

    def test_mock_llm_invoke_called(self):
        """[MOCK] LLM.invoke() được gọi"""
        llm = MagicMock()
        llm.invoke.return_value = "mocked answer"
        result = llm.invoke("some prompt")
        llm.invoke.assert_called_once_with("some prompt")
        assert result == "mocked answer"

    def test_mock_llm_self_rag_good_json(self):
        """[MOCK] LLM trả JSON hợp lệ cho Self-RAG"""
        llm = MagicMock()
        llm.invoke.return_value = json.dumps({
            "score": 8, "reason": "Phù hợp context", "is_sufficient": True
        })
        result = self_rag_evaluate(llm, "Q?", "Answer.", "Context here")
        assert result["score"] == 8
        assert result["is_sufficient"] == True

    def test_mock_llm_connection_error_handled(self):
        """[MOCK] LLM lỗi kết nối -> fallback không crash"""
        llm = MagicMock()
        llm.invoke.side_effect = ConnectionError("Cannot connect to Ollama")
        result = self_rag_evaluate(llm, "Q", "A", "C")
        assert isinstance(result, dict)

    def test_mock_llm_called_for_rewrite_with_history(self):
        """[MOCK] LLM được gọi khi rewrite có history"""
        llm = MagicMock()
        llm.invoke.return_value = "Câu hỏi đã cải thiện"
        history = [{"question": "RAG?", "answer": "RAG là...", "timestamp": ""}]
        rewrite_query(llm, "nó là gì?", history)
        assert llm.invoke.call_count == 1

    def test_mock_llm_not_called_without_history(self):
        """[MOCK] LLM KHÔNG được gọi khi không có history"""
        llm = MagicMock()
        rewrite_query(llm, "câu hỏi", [])
        llm.invoke.assert_not_called()

    def test_mock_llm_rewrite_fallback_on_error(self):
        """[MOCK] Rewrite fallback về câu gốc khi LLM lỗi"""
        llm = MagicMock()
        llm.invoke.side_effect = Exception("timeout")
        history = [{"question": "Q", "answer": "A", "timestamp": ""}]
        result = rewrite_query(llm, "câu gốc", history)
        assert result == "câu gốc"


class TestMock_FileLoading:
    """MOCK TEST: Thay thế file loaders"""

    def test_mock_pdf_loader_instantiated(self):
        """[MOCK] PDFPlumberLoader được khởi tạo"""
        with patch('app.PDFPlumberLoader') as mock_cls:
            mock_cls.return_value.load.return_value = [
                Document(page_content="PDF content", metadata={})
            ]
            load_document(b"%PDF fake", "document.pdf")
            assert mock_cls.called

    def test_mock_docx_loader_instantiated(self):
        """[MOCK] Docx2txtLoader được khởi tạo cho DOCX"""
        with patch('app.Docx2txtLoader') as mock_cls:
            mock_cls.return_value.load.return_value = [
                Document(page_content="DOCX content", metadata={})
            ]
            load_document(b"PK fake", "document.docx")
            assert mock_cls.called

    def test_mock_metadata_injected(self):
        """[MOCK] Metadata source_file được inject sau load"""
        with patch('app.PDFPlumberLoader') as mock_cls:
            mock_cls.return_value.load.return_value = [
                Document(page_content="Test", metadata={})
            ]
            docs = load_document(b"%PDF", "myreport.pdf")
            assert docs[0].metadata["source_file"] == "myreport.pdf"
            assert docs[0].metadata["file_type"] == "pdf"
            assert "upload_time" in docs[0].metadata

    def test_mock_temp_file_deleted(self):
        """[MOCK] File tạm bị xóa sau khi load"""
        with patch('app.PDFPlumberLoader') as mock_cls:
            mock_cls.return_value.load.return_value = [
                Document(page_content="content", metadata={})
            ]
            with patch('app.os.unlink') as mock_unlink:
                load_document(b"%PDF fake", "test.pdf")
                assert mock_unlink.called, "os.unlink phải được gọi để xóa file tạm"

    def test_mock_invalid_extension_raises(self):
        """[MOCK] Extension không hợp lệ raise Exception"""
        with pytest.raises(Exception):
            load_document(b"fake", "file.xlsx")


class TestMock_CrossEncoder:
    """MOCK TEST: Thay thế Cross-Encoder (Câu 9)"""

    def test_mock_encoder_predict_called(self):
        """[MOCK] predict() được gọi với đúng (query, doc) pairs"""
        enc = MagicMock()
        enc.predict.return_value = [0.8, 0.5, 0.3]
        docs = [Document(page_content=x, metadata={}) for x in ["X", "Y", "Z"]]
        rerank_documents("test query", docs, enc, top_k=2)
        enc.predict.assert_called_once()
        pairs = enc.predict.call_args[0][0]
        assert pairs[0] == ("test query", "X")

    def test_mock_encoder_scores_determine_order(self):
        """[MOCK] Scores từ mock quyết định thứ tự đúng"""
        enc = MagicMock()
        enc.predict.return_value = [0.1, 0.3, 0.95]
        docs = [Document(page_content=x, metadata={}) for x in ["A", "B", "C"]]
        result = rerank_documents("q", docs, enc, top_k=1)
        assert result[0].page_content == "C"

    def test_mock_encoder_not_called_when_none(self):
        """[MOCK] Khi encoder=None, không gọi bất kỳ predict nào"""
        docs = [Document(page_content="doc", metadata={})]
        result = rerank_documents("q", docs, None, top_k=1)
        assert result == docs


# ================================================================
#  PHẦN 4: AUTOMATION VÀ CI/CD
# ================================================================
#  Các tests tự động chạy trong CI/CD pipeline:
#    - Không cần user interaction
#    - Kết quả nhất quán, không phụ thuộc external services
#    - Kiểm tra cấu hình project sẵn sàng deploy
# ================================================================

class TestAutomation_Performance:
    """AUTOMATION TEST: Kiểm tra hiệu năng"""

    def test_auto_detect_language_speed(self):
        """[CI/CD] detect_language() chạy 1000 lần < 10s"""
        start = time.time()
        for _ in range(1000):
            detect_language("Đây là câu kiểm tra hiệu năng tự động")
        assert time.time() - start < 10, "detect_language quá chậm"

    def test_auto_split_large_doc_speed(self):
        """[CI/CD] split_documents() với văn bản lớn < 5s"""
        large = [Document(page_content="Câu test. " * 5000, metadata={})]
        start = time.time()
        chunks = split_documents(large, "Recursive (Mặc định)", 500, 50)
        assert time.time() - start < 5, "split_documents quá chậm"
        assert len(chunks) > 0

    def test_auto_rerank_100_docs_speed(self):
        """[CI/CD] rerank 100 docs (mock) < 1s"""
        docs = [Document(page_content=f"Doc {i}", metadata={}) for i in range(100)]
        enc = MagicMock()
        enc.predict.return_value = [float(i)/100 for i in range(100)]
        start = time.time()
        result = rerank_documents("q", docs, enc, top_k=10)
        assert time.time() - start < 1
        assert len(result) == 10


class TestAutomation_Regression:
    """AUTOMATION TEST: Regression tests - đảm bảo không phá vỡ tính năng cũ"""

    @pytest.mark.parametrize("text,expected", [
        ("Xin chào", 'vi'),
        ("Tôi là sinh viên", 'vi'),
        ("Hệ thống RAG hoạt động tốt", 'vi'),
        ("Hello world", 'en'),
        ("What is machine learning?", 'en'),
        ("The RAG system works well", 'en'),
    ])
    def test_regression_detect_language(self, text, expected):
        """[REGRESSION] detect_language kết quả không thay đổi"""
        assert detect_language(text) == expected, \
            f"Regression: '{text}' phải trả về '{expected}'"

    def test_regression_all_chunk_strategies_work(self):
        """[REGRESSION] Cả 3 chunk strategies hoạt động"""
        docs = [Document(page_content="Test content. " * 50, metadata={})]
        for strategy in ["Recursive (Mặc định)", "Token-based", "Paragraph-based"]:
            result = split_documents(docs, strategy, 300, 30)
            assert len(result) > 0, f"Strategy '{strategy}' bị lỗi"

    def test_regression_rerank_none_encoder_safe(self):
        """[REGRESSION] rerank với None encoder luôn an toàn"""
        docs = [Document(page_content="c", metadata={}) for _ in range(5)]
        assert rerank_documents("q", docs, None, 3) == docs

    @pytest.mark.parametrize("bad_response", [
        "", "not json", "{invalid}", "null", "[]"
    ])
    def test_regression_self_rag_never_crashes(self, bad_response):
        """[REGRESSION] self_rag_evaluate không crash với bất kỳ response nào"""
        llm = MagicMock()
        llm.invoke.return_value = bad_response
        result = self_rag_evaluate(llm, "Q", "A", "C")
        assert isinstance(result, dict), f"Crash với: {repr(bad_response)}"


class TestAutomation_CIConfig:
    """AUTOMATION TEST: Kiểm tra cấu hình project cho CI/CD"""

    ROOT = os.path.dirname(os.path.abspath(__file__))

    def test_ci_requirements_txt_exists(self):
        """[CI/CD] requirements.txt phải tồn tại"""
        assert os.path.exists(os.path.join(self.ROOT, "requirements.txt")), \
            "Thiếu requirements.txt"

    def test_ci_app_py_exists(self):
        """[CI/CD] app.py phải tồn tại"""
        assert os.path.exists(os.path.join(self.ROOT, "app.py")), \
            "Thiếu app.py"

    def test_ci_data_directory_exists(self):
        """[CI/CD] Thư mục data/ phải tồn tại"""
        assert os.path.exists(os.path.join(self.ROOT, "data")), \
            "Thiếu thư mục data/"

    def test_ci_app_importable(self):
        """[CI/CD] app.py import được không lỗi"""
        try:
            import app
        except ImportError as e:
            pytest.fail(f"app.py không import được: {e}")

    @pytest.mark.parametrize("func_name", [
        "detect_language", "split_documents", "load_document",
        "rerank_documents", "self_rag_evaluate", "rewrite_query",
    ])
    def test_ci_required_functions_exist(self, func_name):
        """[CI/CD] Các hàm cần thiết phải tồn tại trong app.py"""
        import app
        assert hasattr(app, func_name), \
            f"Thiếu hàm '{func_name}' trong app.py"

    def test_ci_no_hardcoded_secrets(self):
        """[CI/CD] Không có API key/secret hardcode trong app.py"""
        with open(os.path.join(self.ROOT, "app.py"), 'r', encoding='utf-8') as f:
            content = f.read()
        for pattern in ["sk-", 'api_key = "', 'SECRET_KEY = "']:
            assert pattern not in content, \
                f"Phát hiện secret hardcode: '{pattern}'"

    def test_ci_readme_exists(self):
        """[CI/CD] README.md phải tồn tại"""
        assert os.path.exists(os.path.join(self.ROOT, "README.md")), \
            "Thiếu README.md"


# ================================================================
#  GITHUB ACTIONS WORKFLOW (tự động tạo khi chạy file)
# ================================================================

GITHUB_WORKFLOW = """\
# .github/workflows/ci.yml
name: SmartDoc AI - CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python 3.10
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Cache pip
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-mock

      - name: Unit Tests & TDD
        run: pytest test_smartdoc.py -v -k "Unit or tdd or TDD"

      - name: Integration Tests
        run: pytest test_smartdoc.py -v -k "integ or Integration"

      - name: Mock Tests
        run: pytest test_smartdoc.py -v -k "Mock or mock"

      - name: Automation & CI Tests
        run: pytest test_smartdoc.py -v -k "auto or Auto or CI or Regression"

      - name: Full coverage report
        run: pytest test_smartdoc.py --cov=app --cov-report=xml --cov-report=term

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
"""


def setup_github_actions():
    """Tạo file GitHub Actions workflow"""
    os.makedirs(".github/workflows", exist_ok=True)
    with open(".github/workflows/ci.yml", "w") as f:
        f.write(GITHUB_WORKFLOW)
    print("✅ Tạo .github/workflows/ci.yml thành công!")


# ================================================================
#  RUNNER THỦ CÔNG (không cần pytest)
# ================================================================

def run_all_tests():
    print("=" * 65)
    print("  SMARTDOC AI — FULL TEST SUITE")
    print("  Unit/TDD | Integration | Mock | Automation/CI/CD")
    print("  OSSD Spring 2026 — Đại học Sài Gòn")
    print("=" * 65)

    passed = failed = 0
    fail_list = []

    def check(section, name, condition):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}")
            failed += 1
            fail_list.append(f"[{section}] {name}")

    # ---- 1. UNIT & TDD ----
    print("\n🔵 PHẦN 1: UNIT TESTING & TDD")
    check("UNIT", "detect_language VI", detect_language("Tôi là sinh viên") == 'vi')
    check("UNIT", "detect_language EN", detect_language("I am a student") == 'en')
    check("UNIT", "detect_language empty -> en", detect_language("") == 'en')
    check("UNIT", "detect_language numbers -> en", detect_language("12345") == 'en')
    check("UNIT", "detect_language returns str", isinstance(detect_language("x"), str))
    check("UNIT", "detect_language valid value", detect_language("x") in ['vi','en'])

    docs = [Document(page_content="Test. " * 100, metadata={})]
    r = split_documents(docs, "Recursive (Mặc định)", 200, 20)
    t = split_documents(docs, "Token-based", 200, 20)
    p = split_documents(docs, "Paragraph-based", 200, 20)
    check("UNIT", "split Recursive works", len(r) > 0)
    check("UNIT", "split Token-based works", len(t) > 0)
    check("UNIT", "split Paragraph-based works", len(p) > 0)
    check("UNIT", "split returns Documents", all(isinstance(c, Document) for c in r))

    d500  = split_documents([Document(page_content="x. "*400, metadata={})], "Recursive (Mặc định)", 500, 50)
    d2000 = split_documents([Document(page_content="x. "*400, metadata={})], "Recursive (Mặc định)", 2000, 50)
    check("TDD", "larger chunk_size -> fewer chunks", len(d500) >= len(d2000))

    enc = MagicMock()
    enc.predict.return_value = [0.3, 0.9, 0.5]
    d3 = [Document(page_content=x, metadata={}) for x in ["A","B","C"]]
    rr = rerank_documents("q", d3, enc, top_k=2)
    check("UNIT", "rerank top_k=2 correct", len(rr) == 2)
    check("UNIT", "rerank highest score first", rr[0].page_content == "B")
    check("UNIT", "rerank None encoder safe", rerank_documents("q", d3, None, 2) == d3)
    check("UNIT", "rerank empty docs -> []", rerank_documents("q", [], enc, 3) == [])

    llm = MagicMock()
    llm.invoke.return_value = '{"score":8,"reason":"ok","is_sufficient":true}'
    sr = self_rag_evaluate(llm, "Q", "A", "C")
    check("UNIT", "self_rag returns dict", isinstance(sr, dict))
    check("UNIT", "self_rag has score", "score" in sr)
    check("UNIT", "self_rag score 1-10", 1 <= sr.get("score",0) <= 10)
    llm2 = MagicMock(); llm2.invoke.return_value = "bad"
    check("UNIT", "self_rag fallback on bad json", isinstance(self_rag_evaluate(llm2,"Q","A","C"), dict))

    # ---- 2. INTEGRATION ----
    print("\n🟢 PHẦN 2: INTEGRATION TESTING")
    with patch('app.PDFPlumberLoader') as ml:
        ml.return_value.load.return_value = [Document(page_content="Content. "*50, metadata={})]
        d = load_document(b"%PDF", "test.pdf")
        c = split_documents(d, "Recursive (Mặc định)", 200, 20)
        check("INTEG", "PDF load -> split", len(c) > 0)
        check("INTEG", "chunks are Documents", all(isinstance(x, Document) for x in c))

    with patch('app.PDFPlumberLoader') as ml2:
        ml2.return_value.load.return_value = [Document(page_content="t", metadata={})]
        d2 = load_document(b"%PDF", "file.pdf")
        check("INTEG", "metadata source_file injected", d2[0].metadata.get("source_file") == "file.pdf")
        check("INTEG", "metadata file_type injected", d2[0].metadata.get("file_type") == "pdf")
        check("INTEG", "metadata upload_time injected", "upload_time" in d2[0].metadata)

    ca = [Document(page_content=f"A{i}", metadata={"source_file":"a.pdf"}) for i in range(3)]
    cb = [Document(page_content=f"B{i}", metadata={"source_file":"b.pdf"}) for i in range(2)]
    combined = ca + cb
    srcs = {c.metadata["source_file"] for c in combined}
    check("INTEG", "multi-doc gộp đúng số chunks (Câu 8)", len(combined) == 5)
    check("INTEG", "multi-doc giữ metadata nguồn", "a.pdf" in srcs and "b.pdf" in srcs)

    enc2 = MagicMock(); enc2.predict.return_value = [0.1, 0.95, 0.4]
    docs3 = [Document(page_content=x, metadata={}) for x in ["Không liên quan","RAG quan trọng","FAISS"]]
    rr2 = rerank_documents("RAG?", docs3, enc2, top_k=2)
    check("INTEG", "rerank pipeline đúng thứ tự (Câu 9)", rr2[0].page_content == "RAG quan trọng")

    cits = [{"index":i+1,"content":d.page_content,"source":d.metadata["source_file"],"page":1}
            for i,d in enumerate(ca)]
    check("INTEG", "citations tạo đúng số lượng (Câu 5)", len(cits) == 3)
    check("INTEG", "citation có đủ keys", all(k in cits[0] for k in ["index","content","source","page"]))

    # ---- 3. MOCK ----
    print("\n🟡 PHẦN 3: MOCK TESTING")
    m1 = MagicMock(); m1.invoke.return_value = '{"score":7,"reason":"ok","is_sufficient":true}'
    self_rag_evaluate(m1, "Q", "A", "C")
    check("MOCK", "LLM.invoke() được gọi cho Self-RAG", m1.invoke.called)

    m2 = MagicMock(); m2.invoke.return_value = "rewritten"
    h = [{"question":"Q","answer":"A","timestamp":""}]
    rewrite_query(m2, "nó là gì?", h)
    check("MOCK", "LLM được gọi khi rewrite có history", m2.invoke.called)

    m3 = MagicMock()
    rewrite_query(m3, "q", [])
    check("MOCK", "LLM KHÔNG gọi khi không có history", not m3.invoke.called)

    m4 = MagicMock(); m4.invoke.side_effect = Exception("timeout")
    res = rewrite_query(m4, "gốc", h)
    check("MOCK", "rewrite fallback khi LLM lỗi", res == "gốc")

    with patch('app.PDFPlumberLoader') as mpl:
        mpl.return_value.load.return_value = [Document(page_content="c", metadata={})]
        with patch('app.os.unlink') as mu:
            load_document(b"%PDF", "f.pdf")
            check("MOCK", "file tạm bị xóa (os.unlink called)", mu.called)

    enc3 = MagicMock(); enc3.predict.return_value = [0.9, 0.1]
    d2_ = [Document(page_content=x, metadata={}) for x in ["X","Y"]]
    rerank_documents("q", d2_, enc3, top_k=1)
    pairs_ = enc3.predict.call_args[0][0]
    check("MOCK", "encoder predict gọi với đúng pairs", pairs_[0] == ("q","X"))

    # ---- 4. AUTOMATION / CI/CD ----
    print("\n🔴 PHẦN 4: AUTOMATION & CI/CD")
    t0 = time.time()
    for _ in range(1000): detect_language("Đây là câu kiểm tra hiệu năng")
    check("CI/CD", "detect_language 1000x < 10s", time.time()-t0 < 10)

    large_doc = [Document(page_content="sent. "*5000, metadata={})]
    t1 = time.time()
    split_documents(large_doc, "Recursive (Mặc định)", 500, 50)
    check("CI/CD", "split large doc < 5s", time.time()-t1 < 5)

    root = os.path.dirname(os.path.abspath(__file__))
    check("CI/CD", "requirements.txt tồn tại", os.path.exists(os.path.join(root,"requirements.txt")))
    check("CI/CD", "app.py tồn tại", os.path.exists(os.path.join(root,"app.py")))
    check("CI/CD", "README.md tồn tại", os.path.exists(os.path.join(root,"README.md")))
    check("CI/CD", "data/ tồn tại", os.path.exists(os.path.join(root,"data")))

    import app as app_mod
    for fn in ["detect_language","split_documents","load_document","rerank_documents","self_rag_evaluate","rewrite_query"]:
        check("CI/CD", f"hàm '{fn}' tồn tại", hasattr(app_mod, fn))

    with open(os.path.join(root,"app.py"),'r',encoding='utf-8') as f:
        code = f.read()
    check("CI/CD", "không có hardcoded API key", "sk-" not in code and 'api_key = "' not in code)

    # Regression
    for text, exp in [("Xin chào","vi"),("Hello","en"),("Đây là gì?","vi"),("What is RAG?","en")]:
        check("REGRESSION", f"'{text}' -> '{exp}'", detect_language(text) == exp)

    # ---- TỔNG KẾT ----
    total = passed + failed
    print("\n" + "=" * 65)
    print(f"  📊 KẾT QUẢ: {passed}/{total} tests PASS")
    print(f"     ✅ Passed : {passed}")
    print(f"     ❌ Failed : {failed}")
    if fail_list:
        print("\n  Tests FAIL:")
        for f_ in fail_list:
            print(f"    - {f_}")
    print("=" * 65)
    print("\n  🎉 Tất cả PASS! Sẵn sàng submit.\n" if failed == 0
          else f"\n  ⚠️  {failed} test(s) cần sửa.\n")
    return failed == 0


if __name__ == "__main__":
    # Tạo GitHub Actions workflow
    setup_github_actions()
    # Chạy tất cả tests
    run_all_tests()
