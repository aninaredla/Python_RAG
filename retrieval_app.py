import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import streamlit as st
from extract_chunk_embed import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL_ID


RESPONSE_LLM = 'gemini-2.5-flash-lite-preview-06-17'
GENAI_API_KEY = ""
TOP_N_RESULTS = 5


def configure_clients():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    genai.configure(api_key=GENAI_API_KEY)
    llm = genai.GenerativeModel(RESPONSE_LLM)
    return chroma_client, llm


def encode_query(text):
    model = SentenceTransformer(EMBED_MODEL_ID)
    return model.encode([text])[0].tolist()


def build_retrieval_prompt(documents, question):
    return f"""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Context:
{documents}

Question:
{question}
"""


def main():
    st.set_page_config(page_title="Nutritionist Chatbot", layout="centered")
    st.title("RAG Chatbot")
    st.write("Ask a question based on the document.")

    user_query = st.text_input("Enter your question:")

    if user_query:
        with st.spinner("Thinking..."):
            # Set up all external systems
            chroma_client, llm = configure_clients()
            table = chroma_client.get_collection(COLLECTION_NAME)

            # Embed and retrieve
            query_embedding = encode_query(user_query)
            results = table.query(query_embeddings=[query_embedding], n_results=TOP_N_RESULTS)
            context_docs = results["documents"]

            # Generate and display response
            full_prompt = build_retrieval_prompt(context_docs, user_query)
            answer = llm.generate_content(full_prompt).text

        st.markdown("## 💬 Answer")
        st.success(answer)

        with st.expander("🔍 Retrieved Context Chunks"):
            for doc in context_docs:
                st.markdown(f"```text\n{doc[:500]}\n```")


if __name__ == "__main__":
    main()
