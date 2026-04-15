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
    logger.info(f"[GraphRAG] Building graph for {filename} ({len(chunks)} chunks)")

    with driver.session() as session:
        session.run("MATCH (n {source_file: $f}) DETACH DELETE n", f=filename)

        for i, chunk in enumerate(chunks):
            text = chunk.page_content
            entities = extract_entities(text)

            session.run("""
                MERGE (d:Document {chunk_id: $id})
                SET d.content = $content, d.source_file = $file
            """, {"id": f"{filename}_{i}", "content": text[:700], "file": filename})

            for e in entities:
                session.run("""
                    MERGE (e:Entity {name: $name, source_file: $file})
                    WITH e
                    MATCH (d:Document {chunk_id: $id})
                    MERGE (d)-[:CONTAINS]->(e)
                """, {"name": e, "file": filename, "id": f"{filename}_{i}"})

    driver.close()
    logger.info(f"[GraphRAG] Build completed for {filename}")


def query_graph(question: str, filename: str = None, top_k: int = 3):
    driver = get_driver()
    docs = []

    with driver.session() as session:
        # Kiểm tra tổng số Document nodes
        count = session.run("MATCH (d:Document) RETURN count(d) as cnt").single()["cnt"]
        logger.info(f"[GraphRAG] Total Document nodes in Neo4j: {count}")

        if count == 0:
            logger.warning("[GraphRAG] ⚠️ Neo4j is completely empty!")
            driver.close()
            return []

        # FULL FALLBACK - Lấy tất cả documents không cần filter filename
        logger.info("[GraphRAG] Full fallback - Getting all documents")
        result = session.run("""
            MATCH (d:Document)
            RETURN d.content as content, d.page as page
            LIMIT $limit
        """, {"limit": top_k * 6})

        for r in result:
            docs.append(Document(
                page_content=r["content"],
                metadata={"method": "fallback", "page": r.get("page")}
            ))

    driver.close()
    logger.info(f"[GraphRAG] Retrieved {len(docs)} documents (full fallback)")
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