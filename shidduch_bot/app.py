"""
קליינט פשוט ב-Streamlit:
    streamlit run app.py
"""
import streamlit as st

import config
from rag_engine import answer, build_index

st.set_page_config(page_title="בית נאמן - עוזר השידוכים", page_icon="💍")

# תמיכה בכיוון ימין-לשמאל
st.markdown(
    "<style>body, .stMarkdown, .stChatMessage, textarea, input "
    "{direction: rtl; text-align: right;}</style>",
    unsafe_allow_html=True,
)

st.title("💍 בית נאמן - עוזר השידוכים")
st.caption("שאלו על הרשמה, דמי שדכנות, שלבי התהליך ופרטיות. הבוט עונה רק מתוך נהלי המשרד.")


@st.cache_resource(show_spinner="טוען את מאגר המסמכים...")
def get_index():
    return build_index()


with st.sidebar:
    st.header("הגדרות")
    config.QUERY_EXPANSION_ENABLED = st.toggle(
        "הרחבת שאלות קצרות", value=config.QUERY_EXPANSION_ENABLED
    )
    config.TOP_K = st.slider("TOP K", 1, 10, config.TOP_K)
    show_sources = st.checkbox("הצג קטעי מקור", value=True)
    if st.button("בנה אינדקס מחדש"):
        get_index.clear()
        build_index(force_rebuild=True)
        st.success("האינדקס נבנה מחדש")
    if st.button("נקה שיחה"):
        st.session_state.messages = []

try:
    index = get_index()
except Exception as e:
    st.error(f"שגיאה בטעינה: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("הקלידו שאלה..."):
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("מחפש במסמכים..."):
            result = answer(index, question, history)
        st.markdown(result["answer"])
        if show_sources:
            if result["search_query"] != question:
                st.caption(f"השאלה הורחבה ל: {result['search_query']}")
            for s in result["sources"]:
                with st.expander(f"📄 {s['file']} (דמיון: {s['score']})"):
                    st.write(s["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"]}
    )
