"""
src/chunker.py
Câu 4: Cải thiện chunk strategy
Muốn thêm strategy mới? Thêm elif và splitter tương ứng.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from src.logger import logger

STRATEGIES = ["Recursive (Mặc định)", "Token-based", "Paragraph-based"]


def split_documents(docs: list, strategy: str, chunk_size: int, chunk_overlap: int) -> list:
    """
    Chia documents thành chunks theo strategy được chọn.

    Args:
        docs:          Danh sách Document từ load_document()
        strategy:      Tên strategy (xem STRATEGIES)
        chunk_size:    Kích thước tối đa mỗi chunk (ký tự)
        chunk_overlap: Số ký tự trùng lặp giữa các chunk liên tiếp

    Returns:
        Danh sách chunks (Document objects)
    """
    logger.info(f"Chunking | strategy={strategy} | size={chunk_size} | overlap={chunk_overlap}")

    if strategy == "Token-based":
        # Chia theo token — phù hợp với model có giới hạn token
        splitter = TokenTextSplitter(
            chunk_size=chunk_size // 4,
            chunk_overlap=chunk_overlap // 4
        )
    elif strategy == "Paragraph-based":
        # Chia theo đoạn văn — giữ nguyên cấu trúc đoạn
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n"]
        )
    else:
        # Recursive (mặc định) — linh hoạt nhất
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )

    chunks = splitter.split_documents(docs)
    logger.info(f"Split thành {len(chunks)} chunks")
    return chunks
