from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from .config import settings
from .vector_store import get_vectorstore
from .session_manager import get_session_vectorstore, get_session_history

llm = ChatOpenAI(
    base_url=settings.llm_url,
    api_key=settings.llm_api_key,
    model="Qwen3-32B",
    temperature=0.3,
    max_tokens=16000,
)

system_prompt = """Ты — внутренний корпоративный ассистент. Отвечай строго на русском языке.
Используй ТОЛЬКО контекст и историю диалога.
Если информации нет — скажи: "Информация отсутствует в базе знаний."

Контекст:
{context}

История:
{chat_history}

Вопрос: {question}
Ответ:"""

prompt = ChatPromptTemplate.from_template(system_prompt)

def format_docs(docs):
    return "\n\n".join(f"Источник: {d.metadata.get('source','—')}\n{d.page_content}" for d in docs)

def create_chain(session_id: str):
    global_vs = get_vectorstore()
    session_vs = get_session_vectorstore(session_id)
    history = get_session_history(session_id)

    def retrieve(input_str: str):
        query = input_str if isinstance(input_str, str) else input_str.get("question", "")
        docs1 = global_vs.similarity_search(query, k=12)
        docs2 = session_vs.similarity_search(query, k=6) if session_vs else []
        all_docs = docs1 + docs2
        seen = set()
        unique = []
        for d in all_docs:
            key = hash(d.page_content[:300])
            if key not in seen and len(unique) < 15:
                seen.add(key)
                unique.append(d)
        return unique

    def format_history(_):
        msgs = history.chat_memory.messages[-12:]
        return "\n".join(f"{'Пользователь' if m.type=='human' else 'Ассистент'}: {m.content}" for m in msgs) if msgs else "История пуста."

    chain = (
        RunnableLambda(lambda x: x) |
        {
            "context": RunnableLambda(retrieve) | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            "chat_history": RunnableLambda(format_history),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
