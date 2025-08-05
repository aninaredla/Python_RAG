import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import streamlit as st
from extract_chunk_embed import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL_ID
import os
import time


RESPONSE_LLM = 'gemini-2.5-flash-lite-preview-06-17'
GENAI_API_KEY = "AIzaSyAQ2U0t0yX7kMJuKPWTtcbTaYsHBPN0ELQ"
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
    st.set_page_config(page_title="Driver Manual Chatbot", layout="centered")
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
            t1 = time.perf_counter()
            results = table.query(query_embeddings=[query_embedding], n_results=TOP_N_RESULTS)
            t2 = time.perf_counter()
            st.markdown("Chunk Retrieval Time: " + str(t2-t1) + "s")
            context_docs = results["documents"]
            context_metadatas = results["metadatas"]
            context_ids = results["ids"]

            # Generate and display response
            full_prompt = build_retrieval_prompt(context_docs, user_query)
            t3 = time.perf_counter()
            answer = llm.generate_content(full_prompt).text
            t4 = time.perf_counter()
            st.markdown("LLM Response Generation Time: " + str(t4-t3) + "s")

        st.markdown("## 💬 Answer")
        st.success(answer)

        with st.expander("🔍 Retrieved Context Chunks"):
            for i, doc in enumerate(context_docs[0]):
                metadata = context_metadatas[0][i]
                id = context_ids[0][i]
                filename = os.path.basename(metadata.get('source', 'Unknown file'))
                page_number = metadata.get('page_label', metadata.get('page', 0) + 1)

                st.markdown(f"**Source:** {filename} — **Page:** {page_number} — **Chunk ID: ** {id}")
                st.markdown(f"```text\n{doc[:500]}\n```")


if __name__ == "__main__":
    main()
