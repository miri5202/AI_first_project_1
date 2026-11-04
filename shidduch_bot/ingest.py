"""
הרצה חד-פעמית (או אחרי שינוי מסמכים) לבניית האינדקס מחדש:
    python ingest.py
"""
from rag_engine import build_index, _get_collection

if __name__ == "__main__":
    build_index(force_rebuild=True)
    _, collection = _get_collection()
    print(f"האינדקס נבנה בהצלחה. מספר chunks ב-ChromaDB: {collection.count()}")
