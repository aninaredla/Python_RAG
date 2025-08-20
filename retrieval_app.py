import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import streamlit as st
from extract_chunk_embed import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL_ID
import os
import time

RESPONSE_LLM = 'gemini-2.5-flash-lite'
GENAI_API_KEY = ""
TOP_N_RESULTS = 5


@st.cache_resource
def get_clients():
    """Initialize and cache clients."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    genai.configure(api_key=GENAI_API_KEY)
    llm = genai.GenerativeModel(RESPONSE_LLM)
    model = SentenceTransformer(EMBED_MODEL_ID)
    return chroma_client.get_collection(COLLECTION_NAME), llm, model


def build_retrieval_prompt(documents, question):
    return f"""You are an expert driving instructor specializing in traffic laws and safe driving practices for the east coast of the USA. Your knowledge base is strictly limited to the official state driver's manuals provided as context.

<Thought_Process>
Before answering, you must follow these internal steps. This analysis is for your reasoning only and **MUST NOT** be included in your final response.
1.  **Analyze the Question:** Read the `{question}` to infer the implied situation.
2.  **Consider Key Factors:** Silently determine the following:
    -   *Vehicle State:* Is it moving or stationary?
    -   *Driver Location:* Is the driver inside or outside the vehicle?
    -   *Setting:* Is it a public road, a private driveway, a parking lot, etc.?
3.  **Consult the Context:** Review the provided `{documents}` to find the rules that apply to the specific situation you've identified.
</Thought_Process>

<Output_Rules>
1.  **Final Answer Only:** Your response must ONLY contain the direct answer to the question. Do not show your situational analysis. Do not add greetings or conversational filler.
2.  **Strictly Grounded:** Base your answer *exclusively* on the information within the provided `Context`.
3.  **Handle Unknowns:** If the `Context` does not contain the answer, state that the information is not available in the provided manuals.
4.  **Be Concise:** Keep your answer to a maximum of three (3) sentences.
</Output_Rules>

<Task>
Use the `Context` below to answer the `Question`.

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
            collection, llm, model = get_clients()
            
            # Retrieve chunks
            t1 = time.perf_counter()
            query_embedding = model.encode([user_query])[0].tolist()
            results = collection.query(query_embeddings=[query_embedding], n_results=TOP_N_RESULTS)
            t2 = time.perf_counter()
            
            st.markdown(f"Chunk Retrieval Time: {t2-t1:.4f}s.")
            st.markdown(f"Retrieved {len(results['documents'][0])} chunks.")
            
            # Generate response
            t3 = time.perf_counter()
            full_prompt = build_retrieval_prompt(results["documents"][0], user_query)
            answer = llm.generate_content(full_prompt).text
            t4 = time.perf_counter()
            st.markdown(f"LLM Response Generation Time: {t4-t3:.4f}s.")

        st.markdown("## 💬 LLM Answer")
        st.success(answer)

        with st.expander("🔍 Retrieved Context Chunks:"):
            docs, metadatas, ids = results["documents"][0], results["metadatas"][0], results["ids"][0]
            for i, (doc, metadata, chunk_id) in enumerate(zip(docs, metadatas, ids)):
                filename = os.path.basename(metadata.get('source', 'Unknown file'))
                page_number = metadata.get('page_label', metadata.get('page', 0) + 1)
                
                st.markdown(f"**Source:** {filename} — **Page:** {page_number} — **Chunk ID:** {chunk_id}")
                st.markdown(f"```text\n{doc[:500]}\n```")


if __name__ == "__main__":
    main()