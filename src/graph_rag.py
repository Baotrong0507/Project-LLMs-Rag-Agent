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


def build_graph_from_chunks(chunks: list, filename: str):
    driver = get_driver()
    logger.info(f"[GraphRAG] Building/checking graph for {filename} ({len(chunks)} chunks)")

    with driver.session() as session:
        # Kiểm tra graph đã tồn tại chưa (sửa bug: dùng session đúng)
        result = session.run("""
            MATCH (d:Document {source_file: $f})
            RETURN count(d) as count
        """, f=filename)
        
        if result.single()["count"] > 0:
            logger.info(f"[INFO] Graph for {filename} already exists. Skipping build.")
            driver.close()
            return

        # Xóa cũ (nếu cần)
        session.run("MATCH (n {source_file: $f}) DETACH DELETE n", f=filename)

        for i, chunk in enumerate(chunks):
            text = chunk.page_content
            entities = extract_entities(text)

            # Tạo Document node
            chunk_id = f"{filename}_{i}"
            session.run("""
                MERGE (d:Document {chunk_id: $id})
                SET d.content = $content, 
                    d.source_file = $file,
                    d.chunk_index = $idx
            """, {"id": chunk_id, "content": text[:1000], "file": filename, "idx": i})

            # Tạo Entity và relationship
            for e in entities:
                session.run("""
                    MERGE (e:Entity {name: $name, source_file: $file})
                    MERGE (d:Document {chunk_id: $id})-[:CONTAINS]->(e)
                """, {"name": e, "file": filename, "id": chunk_id})

    driver.close()
    logger.info(f"[GraphRAG] Build completed for {filename}")


def query_graph(question: str, filename: str = None, top_k: int = 3):
    driver = get_driver()
    docs = []

    with driver.session() as session:
        count = session.run("MATCH (d:Document) RETURN count(d) as cnt").single()["cnt"]
        logger.info(f"[GraphRAG] Total Document nodes in Neo4j: {count}")

        if count == 0:
            logger.warning("[GraphRAG] ⚠️ Neo4j is empty!")
            driver.close()
            return []

        # === CẢI TIẾN: Tìm theo Entity liên quan đến question ===
        logger.info(f"[GraphRAG] Searching graph for question: {question[:100]}...")

        result = session.run("""
            // Tìm entities khớp với từ trong question (text search)
            CALL db.index.fulltext.queryNodes("entityNameIndex", $query) YIELD node, score
            WITH node as e, score
            MATCH (d:Document)-[:CONTAINS]->(e)
            WHERE ($filename IS NULL OR d.source_file = $filename)
            RETURN DISTINCT d.content as content, d.chunk_index as page, score
            ORDER BY score DESC
            LIMIT $limit
        """, {
            "query": question, 
            "filename": filename,
            "limit": top_k * 5
        })

        for r in result:
            docs.append(Document(
                page_content=r["content"],
                metadata={
                    "method": "graph_entity",
                    "score": r["score"],
                    "page": r.get("page")
                }
            ))

        # Nếu không tìm được gì → fallback nhẹ
        if len(docs) == 0:
            logger.info("[GraphRAG] No graph match → light fallback")
            result = session.run("""
                MATCH (d:Document)
                WHERE ($filename IS NULL OR d.source_file = $filename)
                RETURN d.content as content, d.chunk_index as page
                LIMIT $limit
            """, {"filename": filename, "limit": top_k * 3})
            
            for r in result:
                docs.append(Document(page_content=r["content"], metadata={"method": "fallback"}))

    driver.close()
    logger.info(f"[GraphRAG] Retrieved {len(docs)} documents via graph")
    return docs[:top_k]


def graph_retriever_for_file(question: str, filename: str = None, top_k: int = 3):
    try:
        return query_graph(question, filename, top_k)
    except Exception as e:
        logger.error(f"GraphRAG error: {e}")
        return []


# ========================
# HÀM XÓA GRAPH (mới thêm)
# ========================
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