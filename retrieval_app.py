import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import streamlit as st
# from extract_chunk_embed import embedding_function

CHROMA_PATH = "./chroma_db"
RESPONSE_LLM = 'gemini-2.5-flash-lite-preview-06-17'
GENAI_API_KEY = ""
TOP_N_RESULTS = 5
EMBED_MODEL_ID = "all-MiniLM-L6-v2"


# --- PART 1: SETUP --- #
# chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)    # initialize ChromaDB path
# genai.configure(api_key=GENAI_API_KEY)    # configure google gemini REMEMBER TO REMOVE KEY
# table = chroma_client.get_collection("docling_chunks")
# llm = genai.GenerativeModel(RESPONSE_LLM)

def configure_clients():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    genai.configure(api_key=GENAI_API_KEY)
    llm = genai.GenerativeModel(RESPONSE_LLM)
    return chroma_client, llm


def load_vector_collection(client, collection_name="docling_chunks"):
    return client.get_collection(collection_name)


def encode_query(text):
    model = SentenceTransformer(EMBED_MODEL_ID)
    return model.encode([text])[0].tolist()


def retrieve_relevant_chunks(table, query_embedding, top_k):
    return table.query(query_embeddings=[query_embedding], n_results=top_k)


def build_prompt(documents, question):
    return f"""
You are a helpful assistant. You must only use information provided to answer the question.
If an answer can be reasonably inferred from the provided information, then answer accordingly.
If you are unable to answer, say "I don't know" and state what kind of information/chunks you need.
Answer as concisely as possible without losing key details.

Extracted Information:
{documents}

Question:
{question}

Answer:
"""


def generate_response(llm, prompt):
    return llm.generate_content(prompt).text


def main():
    st.set_page_config(page_title="Nutritionist Chatbot", layout="centered")
    st.title("RAG Chatbot")
    st.write("Ask a question based on the document!")

    user_query = st.text_input("Enter your question:")

    if user_query:
        with st.spinner("Thinking..."):
            # Set up all external systems
            chroma_client, llm = configure_clients()
            table = load_vector_collection(chroma_client)

            # Embed and retrieve
            query_embedding = encode_query(user_query)
            results = retrieve_relevant_chunks(table, query_embedding, TOP_N_RESULTS)
            context_docs = results["documents"]

            # Generate and display response
            full_prompt = build_prompt(context_docs, user_query)
            answer = generate_response(llm, full_prompt)

        st.markdown("## 💬 Answer")
        st.success(answer)

        with st.expander("🔍 Retrieved Context Chunks"):
            for doc in context_docs:
                st.markdown(f"```text\n{doc[:500]}\n```")


if __name__ == "__main__":
    main()


# # ---- Streamlit UI ----
# st.set_page_config(page_title="Nutritionist Chatbot", layout="centered")
# st.title("🥗 Nutritionist RAG Chatbot")
# st.write("Ask a question based on the document!")

# user_query = st.text_input("Enter your question:")

# if user_query:

#     embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#     query_embedding = embedding_model.encode([user_query]).tolist()[0]


#     results = table.query(query_embeddings=[query_embedding], n_results=5)
#     # results = table.query(query_texts=[user_query], n_results=5)

#     full_prompt = f"""
# You are an expert nutritionist assistant. You must only use information provided to answer questions.
# If you are unable to answer, say "I don't know" and state what kind of information/chunks you need.

# Extracted Data:
# {results['documents']}

# Question:
# {user_query}

# Answer:
# """
#     st.markdown("## 💬 Answer")
#     with st.spinner("Thinking..."):
#         response = llm.generate_content(full_prompt)
#         st.success(response.text)
#     with st.expander("🔍 Retrieved Context Chunks"):
#         for block in results["documents"]:
#             st.markdown(f"```text\n{block[:500]}\n```")
