import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# Page settings
st.set_page_config(
    page_title="Mini RAG Q&A",
    page_icon="📚"
)

st.title("📚 Mini RAG Q&A")
st.write("Add a document and ask questions about it.")

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ChromaDB
client = chromadb.Client()

collection = client.get_or_create_collection(
    name="documents"
)

# Document input
document = st.text_area(
    "📄 Paste your document here",
    height=250
)

# Add document
if st.button("➕ Add Document"):

    if not document.strip():
        st.warning("Please enter a document.")

    else:
        chunks = [
            document[i:i + 500]
            for i in range(0, len(document), 500)
        ]

        embeddings = model.encode(chunks)

        collection.add(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings.tolist()
        )

        st.success(
            f"Added {len(chunks)} document chunk(s)!"
        )

# Question
question = st.text_input(
    "❓ Ask a question"
)

# Ask AI
if st.button("🔍 Ask AI"):

    if not question.strip():
        st.warning("Please enter a question.")

    elif collection.count() == 0:
        st.warning("Please add a document first.")

    else:
        question_embedding = model.encode(
            [question]
        )[0]

        results = collection.query(
            query_embeddings=[
                question_embedding.tolist()
            ],
            n_results=min(3, collection.count())
        )

        retrieved_chunks = results["documents"][0]

        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
You are a helpful AI assistant.

Answer the question only using the document below.

Document:
{context}

Question:
{question}

If the answer is not in the document, say:
"I don't know based on the provided document."
"""

        try:
            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            st.subheader("🤖 Answer")

            st.write(
                response["message"]["content"]
            )

        except Exception as e:

            st.error(
                "Ollama is not running or llama3.2 is not installed."
            )

            st.code(str(e))

st.divider()

st.caption(
    "Python + Sentence Transformers + ChromaDB + Ollama + Streamlit"
)