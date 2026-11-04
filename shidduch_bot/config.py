import truststore
truststore.inject_into_ssl()
import os
from dotenv import load_dotenv

load_dotenv()

# --- החלטה 5: ספק API ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- החלטה 7: מודלים ---
# מודל וקטורים: זול מאוד, תומך בעברית, איכות מספיקה לידע ארגוני
EMBEDDING_MODEL = "text-embedding-3-small"
# מודל תשובה: מהיר וזול, מספיק טוב לתשובות מבוססות מסמכים
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0  # 0 = תשובות עקביות, בלי "יצירתיות" שעלולה להמציא

# --- החלטה 1: Vector DB ---
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "shidduch_office_docs"
DATA_DIR = "./data"

# --- החלטה 3: אופן החלוקה (Chunking) לפי כמות טוקנים ---
CHUNK_SIZE = 512  # טוקנים

# --- החלטה 4: חפיפה בין chunks ---
CHUNK_OVERLAP = 64  # טוקנים (כ-12%)

# --- החלטה 6: TOP K ---
TOP_K = 4

# --- החלטה 9: צורת חיפוש ---
# Cosine similarity, עם סף מינימלי: קטעים רחוקים מדי לא יוכנסו להקשר
SIMILARITY_METRIC = "cosine"
SIMILARITY_CUTOFF = 0.30

# --- החלטה 8: הרחבת שאלה ---
# מרחיבים רק שאלות קצרות מאוד (פחות מ-X מילים), כדי לא לשלם על קריאה נוספת לכל שאלה
QUERY_EXPANSION_ENABLED = True
QUERY_EXPANSION_MAX_WORDS = 5
