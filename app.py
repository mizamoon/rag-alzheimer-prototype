from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
from openai import OpenAI
import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1")

USER_PROMPT = """You are a scientific assistant.

Your task is to answer the question using ONLY the information from the context below.

STRICT RULES:
- Use ONLY facts explicitly stated in the context.
- DO NOT use any external knowledge.
- DO NOT make assumptions or generalizations.
- DO NOT add any information that is not directly supported by the context.
- If the context does not contain the answer, say exactly: "The provided context does not contain this information."
- Every statement in your answer MUST be directly supported by the context.

Context:
{context}

Question:
{question}

Answer:
"""

SYSTEM_PROMPT = """You are a scientific assistant.
Follow all rules strictly.
Never use external knowledge.
"""

@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True)
    return db

db = load_db()

def rag_answer(query, db, client, k, temperature):
    retriever = db.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)])

    prompt_text = USER_PROMPT.format(
        context=context,
        question=query)

    response = client.chat.completions.create(
        model="xiaomi/mimo-v2-flash:free",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        temperature=temperature)

    answer = response.choices[0].message.content.strip()

    return answer, docs

st.set_page_config(
    page_title="Прототип RAG-агента",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.sidebar.title("⚙️ Настройки")

k = st.sidebar.slider(
    "Сколько абстрактов брать (k)",
    min_value=1,
    max_value=10,
    value=5,
    step=1
)

temperature = st.sidebar.slider(
    "Температура модели",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05
)

st.markdown(
    "<h1 style='text-align: center;'>🔬 Прототип RAG-агента для поиска терапевтических мишеней при болезни Альцгеймера</h1>",
    unsafe_allow_html=True
)
st.divider()
st.markdown("""
### 🧠 О проекте

Это прототип RAG-ассистента для работы с научной литературой по болезни Альцгеймера. Приложение выполняет семантический поиск по базе научных аннотаций и генерирует ответы строго на основе найденных источников.

Система предназначена для помощи исследователям в анализе публикаций и поиске потенциальных терапевтических мишеней для разработки лекарств.

Также слева, нажав на слайдер, вы можете редактировать настройки модели (температура, количество абстрактов)
""")

st.divider()
query = st.text_area("Введите вопрос:")

if st.button("Найти ответ"):
    if not query.strip():
        st.warning("Введите вопрос")
    else:
        with st.spinner("Думаю..."):
            answer, docs = rag_answer(query, db, client, k, temperature)

        st.subheader("🧠 Ответ:")
        st.markdown(answer)
        st.subheader("📚 Источники:")

        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            title = meta.get("title", "No title")
            year = meta.get("year", "")
            authors = meta.get("authors", "")

            with st.expander(f"[{i}] {title} ({year})"):
                st.write(authors)
                st.write(doc.page_content)