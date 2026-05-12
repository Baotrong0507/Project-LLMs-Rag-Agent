"""
src/graph_rag.py - FINAL VERSION (có clear_graph)
"""
import re
from neo4j import GraphDatabase
from langchain_core.documents import Document
from src.logger import logger

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password123"

def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def extract_entities(text: str) -> list:
    entities = []
    pattern = r'\b[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ][a-zàáâãèéêìíòóôõùúýăđơư]+\b'
    entities.extend(re.findall(pattern, text))

    tech = re.findall(r'\b(?:RAG|LLM|API|RESTful|GraphRAG|Neo4j|LangChain|Ollama|FAISS)\b', text, re.I)
    entities.extend(tech)

    if len(entities) < 5:
        words = [w.strip() for w in text.split() if len(w) > 4]
        entities.extend(words[:15])

    seen = set()
    return [x for x in entities if len(x) > 2 and not (x in seen or seen.add(x))][:20]


def ensure_fulltext_index(session):
    """Tạo fulltext index cho Entity.name nếu chưa tồn tại."""
    try:
        session.run("""
            CREATE FULLTEXT INDEX entityNameIndex IF NOT EXISTS
            FOR (e:Entity) ON EACH [e.name]
        """)
    except Exception as e:
        logger.warning(f"[GraphRAG] Could not create fulltext index: {e}")


def build_graph_from_chunks(chunks: list, filename: str):
    driver = get_driver()
    logger.info(f"[GraphRAG] Building/checking graph for {filename} ({len(chunks)} chunks)")

    with driver.session() as session:
        ensure_fulltext_index(session)

        result = session.run(
            "MATCH (d:Document {source_file: $f}) RETURN count(d) as count",
            f=filename
        )
        if result.single()["count"] > 0:
            logger.info(f"[GraphRAG] Graph for {filename} already exists. Skipping build.")
            driver.close()
            return

        session.run("MATCH (n {source_file: $f}) DETACH DELETE n", f=filename)
        logger.info(f"[GraphRAG] Cleared old data for {filename}, building fresh...")

        for i, chunk in enumerate(chunks):
            text     = chunk.page_content
            entities = extract_entities(text)
            chunk_id = f"{filename}_{i}"

            try:
                session.run("""
                    CREATE (d:Document {
                        chunk_id:    $id,
                        content:     $content,
                        source_file: $file,
                        chunk_index: $idx
                    })
                """, {"id": chunk_id, "content": text[:1000], "file": filename, "idx": i})
            except Exception as e:
                logger.warning(f"[GraphRAG] Skip chunk {i} (conflict): {e}")
                continue

            for ent in entities:
                try:
                    session.run("""
                        MERGE (e:Entity {name: $name, source_file: $file})
                        WITH e
                        MATCH (d:Document {chunk_id: $id})
                        MERGE (d)-[:CONTAINS]->(e)
                    """, {"name": ent, "file": filename, "id": chunk_id})
                except Exception as ex:
                    logger.warning(f"[GraphRAG] Skip entity '{ent}': {ex}")

    driver.close()
    logger.info(f"[GraphRAG] Build completed for {filename}")


# ========================
# Stop words — chỉ giữ lại từ thực sự vô nghĩa
# Bỏ: tóm, tắt, nội, dung, tài, liệu, thông, tin (những từ này
# có thể là keyword hữu ích tuỳ ngữ cảnh)
# ========================
STOP_WORDS = {
    # Tiếng Việt — liên từ, giới từ, đại từ, trợ từ
    "của", "và", "hoặc", "các", "trong", "theo", "được",
    "dựa", "trên", "đã", "liệt", "kê", "mục", "lục",
    "văn", "bản", "là", "có", "với", "về", "này",
    "cho", "từ", "một", "những", "để", "như", "khi",
    "sau", "trước", "rằng", "vì", "nên", "mà", "thì",
    "cũng", "đây", "đó", "kia", "họ", "tôi", "bạn",
    # Tiếng Anh — articles, prepositions, conjunctions
    "the", "of", "in", "to", "a", "an", "is", "are",
    "was", "were", "be", "been", "has", "have", "had",
    "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "can", "this", "that", "these",
    "those", "it", "its", "for", "on", "at", "by", "or",
    "and", "but", "not", "with", "from", "as", "if", "so",
}


def _detect_lang(text: str) -> str:
    """Phát hiện ngôn ngữ chính của câu hỏi."""
    vi_chars = set("àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")
    vi_count = sum(1 for c in text.lower() if c in vi_chars)
    return "vi" if vi_count > 1 else "en"


def _extract_keywords(question: str) -> list:
    """
    Tách từ khóa từ câu hỏi.
    Fix 2 vấn đề:
      1. Bỏ stop_words quá rộng (tóm, tắt, nội, dung...)
      2. Nếu câu hỏi tiếng Việt có từ tiếng Anh ở đầu,
         vẫn trả lời bằng tiếng Việt (xử lý ở rag_engine,
         nhưng keyword extraction phải giữ lại từ kỹ thuật)
    """
    lang = _detect_lang(question)

    raw_words = question.split()
    keywords  = []
    for w in raw_words:
        w_clean = w.strip(".,?!:;\"'").lower()
        if len(w_clean) < 2:
            continue
        if w_clean in STOP_WORDS:
            continue
        # Giữ lại từ kỹ thuật dù ngắn (api, llm, rag...)
        tech_terms = {"rag", "llm", "api", "ui", "ux", "db", "ml", "ai", "nlp"}
        if len(w_clean) < 3 and w_clean not in tech_terms:
            continue
        keywords.append(w_clean)

    # Bổ sung: giữ lại từ viết hoa (tên riêng, công nghệ)
    proper = re.findall(
        r'\b[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ][a-zàáâãèéêìíòóôõùúýăđơư]{1,}\b',
        question
    )
    for p in proper:
        pl = p.lower()
        if pl not in STOP_WORDS and pl not in keywords:
            keywords.append(pl)

    keywords = list(dict.fromkeys(keywords))[:10]
    logger.info(f"[GraphRAG] Lang={lang} | Keywords extracted: {keywords}")
    return keywords


def query_graph(question: str, filename: str = None, top_k: int = 3):
    driver = get_driver()
    docs   = []

    with driver.session() as session:
        count = session.run("MATCH (d:Document) RETURN count(d) as cnt").single()["cnt"]
        logger.info(f"[GraphRAG] Total Document nodes in Neo4j: {count}")

        if count == 0:
            logger.warning("[GraphRAG] Neo4j is empty!")
            driver.close()
            return []

        logger.info(f"[GraphRAG] Searching graph for question: {question[:100]}...")

        # Bước 1: tách keyword (đã fix stop_words + proper nouns)
        keywords = _extract_keywords(question)

        # Bước 2: fulltext search từng keyword
        seen_ids = set()
        for kw in keywords:
            try:
                result = session.run("""
                    CALL db.index.fulltext.queryNodes("entityNameIndex", $query)
                    YIELD node, score
                    MATCH (d:Document)-[:CONTAINS]->(node)
                    WHERE ($filename IS NULL OR d.source_file = $filename)
                    AND NOT d.chunk_id IN $seen
                    RETURN DISTINCT d.chunk_id  as chunk_id,
                                    d.content   as content,
                                    d.chunk_index as page,
                                    score
                    ORDER BY score DESC
                    LIMIT $limit
                """, {
                    "query":    kw,
                    "filename": filename,
                    "seen":     list(seen_ids),
                    "limit":    top_k * 2,
                })
                for r in result:
                    cid = r["chunk_id"]
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        docs.append(Document(
                            page_content=r["content"],
                            metadata={
                                "method":  "graph_entity",
                                "keyword": kw,
                                "score":   r["score"],
                                "page":    r.get("page"),
                            }
                        ))
                if len(docs) >= top_k * 3:
                    break
            except Exception as e:
                logger.warning(f"[GraphRAG] Fulltext '{kw}' failed: {e}")

        # Bước 3: fallback nếu thiếu kết quả
        if len(docs) < top_k:
            logger.info(f"[GraphRAG] Only {len(docs)} docs via entity → fallback by file")
            result = session.run("""
                MATCH (d:Document)
                WHERE ($filename IS NULL OR d.source_file = $filename)
                AND NOT d.chunk_id IN $seen
                RETURN d.chunk_id as chunk_id, d.content as content, d.chunk_index as page
                ORDER BY d.chunk_index ASC
                LIMIT $limit
            """, {
                "filename": filename,
                "seen":     list(seen_ids),
                "limit":    (top_k - len(docs)) * 2,
            })
            for r in result:
                docs.append(Document(
                    page_content=r["content"],
                    metadata={"method": "fallback", "page": r.get("page")}
                ))

    driver.close()
    logger.info(f"[GraphRAG] Retrieved {len(docs)} documents via graph")
    return docs[:top_k]


def graph_retriever_for_file(question: str, filename: str = None, top_k: int = 3):
    try:
        return query_graph(question, filename, top_k)
    except Exception as e:
        logger.error(f"GraphRAG error: {e}")
        return []


def clear_graph(filename: str = None):
    """Xóa graph của một file hoặc toàn bộ graph"""
    driver = get_driver()
    with driver.session() as session:
        if filename:
            session.run("MATCH (n {source_file: $f}) DETACH DELETE n", f=filename)
            logger.info(f"Graph cleared for file: {filename}")
        else:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Entire Graph cleared")
    driver.close()