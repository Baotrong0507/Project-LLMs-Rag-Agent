"""
src/rag_engine.py
Câu 6:  Conversational RAG
Câu 10: Self-RAG Evaluation + Query Rewriting
7.2.3:  Muốn đổi LLM? Sửa LLM_MODEL ở đây
"""
import json
import time
from langchain_ollama import OllamaLLM as Ollama
from src.logger import logger
from src.retriever import rerank_documents, load_cross_encoder

# 7.2.3: Đổi model tại đây (sau khi ollama pull <model>)
LLM_MODEL   = "qwen2.5:7b"
TEMPERATURE = 0.7


def detect_language(text: str) -> str:
    """Phát hiện ngôn ngữ đơn giản dựa trên ký tự tiếng Việt"""
    vi_chars = 'àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ'
    return 'vi' if sum(1 for c in text.lower() if c in vi_chars) > 2 else 'en'


def rewrite_query(llm, question: str, chat_history: list) -> str:
    """
    Câu 10: Tự động viết lại câu hỏi dựa trên lịch sử hội thoại
    để tham chiếu đúng ngữ cảnh.
    """
    if not chat_history:
        return question

    history_text = "\n".join([
        f"User: {h['question']}\nBot: {h['answer'][:80]}"
        for h in chat_history[-2:]
    ])
    prompt = f"""Dựa vào lịch sử, viết lại câu hỏi cho rõ ràng hơn.
Chỉ trả lời câu hỏi đã viết lại.

Lịch sử:
{history_text}

Câu hỏi gốc: {question}
Câu hỏi viết lại:"""
    try:
        rewritten = llm.invoke(prompt).strip()
        if 5 < len(rewritten) < 300:
            logger.info(f"Query rewritten: {rewritten}")
            return rewritten
    except Exception:
        pass
    return question


def self_rag_evaluate(llm, question: str, answer: str, context: str) -> dict:
    """
    Câu 10: LLM tự đánh giá chất lượng câu trả lời.
    Trả về dict: {score, reason, is_sufficient}
    """
    eval_prompt = f"""Đánh giá câu trả lời dựa trên ngữ cảnh.
Trả lời CHỈ bằng JSON: {{"score": 1-10, "reason": "lý do", "is_sufficient": true/false}}

Ngữ cảnh: {context[:400]}
Câu hỏi: {question}
Câu trả lời: {answer}

JSON:"""
    try:
        result = llm.invoke(eval_prompt)
        start = result.find("{")
        end   = result.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(result[start:end])
    except Exception:
        pass
    return {"score": 7, "reason": "Không thể đánh giá tự động", "is_sufficient": True}


def get_answer(
    question: str,
    retriever,
    use_rerank: bool,
    use_self_rag: bool,
    use_conversational: bool,
    top_k: int,
    chat_history: list
) -> tuple:
    """
    Pipeline RAG chính:
    1. Query rewriting (Câu 10)
    2. Retrieval
    3. Re-ranking (Câu 9)
    4. Build prompt với history (Câu 6)
    5. Generate answer
    6. Self-RAG evaluation (Câu 10)
    7. Citation tracking (Câu 5)

    Returns: (answer, citations, self_eval, rewritten_q, elapsed)
    """
    logger.info(f"Query: {question}")

    # 7.2.3: Khởi tạo LLM
    llm  = Ollama(model=LLM_MODEL, temperature=TEMPERATURE)
    lang = detect_language(question)
    logger.info(f"Language detected: {lang}")

    # Câu 10: Query rewriting
    rewritten_q = question
    if use_conversational and chat_history:
        rewritten_q = rewrite_query(llm, question, chat_history)

    # Retrieval
    relevant_docs = retriever.invoke(rewritten_q)
    logger.info(f"Retrieved {len(relevant_docs)} documents")

    # Câu 9: Re-ranking
    if use_rerank:
        cross_encoder = load_cross_encoder()
        relevant_docs = rerank_documents(rewritten_q, relevant_docs, cross_encoder, top_k)

    # Câu 5: Citation tracking
    citations = [
        {
            "index":   i + 1,
            "content": doc.page_content[:200],
            "source":  doc.metadata.get("source_file", "Unknown"),
            "page":    doc.metadata.get("page", "N/A"),
        }
        for i, doc in enumerate(relevant_docs)
    ]

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Câu 6: Conversational history
    history_text = ""
    if use_conversational and chat_history:
        history_text = "\n".join([
            f"User: {h['question']}\nBot: {h['answer']}"
            for h in chat_history[-3:]
        ])

    # Build prompt theo ngôn ngữ
    if lang == 'vi':
        prompt = f"""{'Lịch sử hội thoại:' + chr(10) + history_text + chr(10) if history_text else ''}Ngữ cảnh: {context}

Câu hỏi: {question}
Trả lời ngắn gọn bằng tiếng Việt (3-4 câu). Nếu không biết hãy nói không biết.
Trả lời:"""
    else:
        prompt = f"""{'Conversation history:' + chr(10) + history_text + chr(10) if history_text else ''}Context: {context}

Question: {question}
Answer concisely in English (3-4 sentences). If unsure, say so.
Answer:"""

    # Generate
    t0      = time.time()
    answer  = llm.invoke(prompt)
    elapsed = time.time() - t0
    logger.info(f"Answer generated in {elapsed:.1f}s")

    # Câu 10: Self-RAG evaluation
    self_eval = None
    if use_self_rag:
        self_eval = self_rag_evaluate(llm, question, answer, context)
        logger.info(f"Self-RAG score: {self_eval.get('score', 'N/A')}/10")

    return answer.strip(), citations, self_eval, rewritten_q, elapsed
