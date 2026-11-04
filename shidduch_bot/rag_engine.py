"""
מנוע ה-RAG: טעינת מסמכים, יצירת אינדקס ב-ChromaDB, חיפוש ויצירת תשובה.
"""
import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

NO_ANSWER = "מצטער, לא מצאתי מידע על כך במסמכי המשרד. לפרטים נוספים ניתן לפנות למזכירות בטלפון 03-6188420."

SYSTEM_PROMPT = f"""אתה העוזר הדיגיטלי של משרד השידוכים "בית נאמן". אתה עונה למשפחות על נהלי המשרד והתהליך. עליך לענות אך ורק על סמך קטעי המסמכים שיסופקו לך.
כללים:
1. אסור להשתמש בידע כללי או להמציא מידע.
2. אם התשובה לא מופיעה בקטעים, ענה בדיוק: "{NO_ANSWER}"
3. ענה בעברית, בקצרה ובבהירות.
4. בסוף התשובה ציין מאיזה מסמך נלקח המידע.
5. אינך מכריע בשאלות הלכתיות או השקפתיות. בשאלות כאלה הפנה לרב המשפחה.
6. דבר בלשון מכובדת ועדינה."""

EXPANSION_PROMPT = """הלקוח שאל שאלה קצרה במערכת שאלות על נהלי משרד שידוכים
(הרשמה, דמי הרשמה, דמי שדכנות, שלבי ההצעה, פגישות, פרטיות, שעות פעילות וכו').
נסח אותה מחדש כשאלה מלאה וברורה במשפט אחד, בעברית, בלי להוסיף עובדות.
החזר רק את השאלה המנוסחת.

שאלה: {question}"""


def _init_settings():
    if not config.OPENAI_API_KEY:
        raise RuntimeError("חסר OPENAI_API_KEY בקובץ .env")
    Settings.embed_model = OpenAIEmbedding(
        model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY
    )
    Settings.llm = OpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.OPENAI_API_KEY,
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )


def _get_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return client, client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": config.SIMILARITY_METRIC},
    )


def build_index(force_rebuild: bool = False) -> VectorStoreIndex:
    """בונה אינדקס מהתיקייה data, או טוען אינדקס קיים מ-ChromaDB."""
    _init_settings()
    client, collection = _get_collection()

    if force_rebuild and collection.count() > 0:
        client.delete_collection(config.COLLECTION_NAME)
        client, collection = _get_collection()

    vector_store = ChromaVectorStore(chroma_collection=collection)

    # אם כבר יש וקטורים - לא מחשבים שוב (חוסך כסף ומונע כפילויות)
    if collection.count() > 0:
        return VectorStoreIndex.from_vector_store(vector_store)

    documents = SimpleDirectoryReader(
        config.DATA_DIR, required_exts=[".txt", ".md", ".pdf", ".docx"]
    ).load_data()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )


def expand_question(question: str) -> str:
    """הרחבת שאלה קצרה מדי לפני יצירת הווקטור."""
    if not config.QUERY_EXPANSION_ENABLED:
        return question
    if len(question.split()) >= config.QUERY_EXPANSION_MAX_WORDS:
        return question
    response = Settings.llm.complete(EXPANSION_PROMPT.format(question=question))
    expanded = response.text.strip()
    return expanded or question


def answer(index: VectorStoreIndex, question: str, history=None) -> dict:
    """מחזיר תשובה + הקטעים שנמצאו (לשקיפות)."""
    search_query = expand_question(question)

    retriever = index.as_retriever(similarity_top_k=config.TOP_K)
    nodes = retriever.retrieve(search_query)
    nodes = [n for n in nodes if (n.score or 0) >= config.SIMILARITY_CUTOFF]

    if not nodes:
        return {"answer": NO_ANSWER, "search_query": search_query, "sources": []}

    context = "\n\n---\n\n".join(
        f"[מסמך: {n.metadata.get('file_name', 'לא ידוע')}]\n{n.get_content()}"
        for n in nodes
    )

    messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
    for turn in (history or [])[-4:]:  # מעט היסטוריה לשאלות המשך
        messages.append(ChatMessage(role=turn["role"], content=turn["content"]))
    messages.append(
        ChatMessage(
            role="user",
            content=f"קטעי מסמכים:\n{context}\n\nשאלה: {question}",
        )
    )

    response = Settings.llm.chat(messages)
    return {
        "answer": response.message.content.strip(),
        "search_query": search_query,
        "sources": [
            {
                "file": n.metadata.get("file_name", "לא ידוע"),
                "score": round(n.score or 0, 3),
                "text": n.get_content(),
            }
            for n in nodes
        ],
    }
