"""
src/document_loader.py
Câu 1: Hỗ trợ load file PDF và DOCX
Muốn thêm định dạng mới? Chỉ cần thêm elif ở đây.
"""
import os
import tempfile
from datetime import datetime

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import Docx2txtLoader

from src.logger import logger


def load_document(file_bytes: bytes, filename: str) -> list:
    """
    Load file PDF hoặc DOCX thành danh sách Document.
    Tự động gắn metadata: source_file, file_type, upload_time.
    """
    logger.info(f"Loading file: {filename}")
    ext = filename.lower().split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if ext == "pdf":
            loader = PDFPlumberLoader(tmp_path)
        elif ext == "docx":
            loader = Docx2txtLoader(tmp_path)
        else:
            raise ValueError(f"Định dạng '{ext}' không được hỗ trợ! Chỉ hỗ trợ PDF và DOCX.")

        docs = loader.load()

        # Gắn metadata cho Câu 8 (multi-document filtering)
        for doc in docs:
            doc.metadata["source_file"] = filename
            doc.metadata["file_type"]   = ext
            doc.metadata["upload_time"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        logger.info(f"Loaded {len(docs)} pages from {filename}")
        return docs

    finally:
        os.unlink(tmp_path)
