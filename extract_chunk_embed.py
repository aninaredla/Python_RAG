from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from semantic_chunker.core import SemanticChunker

# Setup
PDF_PATH = "FE1164.pdf"
MAX_TOKENS = 256
TOKEN_OVERLAP = 64
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "docling_chunks"


def main():

    if os.path.exists(CHROMA_PATH):
        clear_db = input("Clear and re-create database (y/n)? :  ")
        if clear_db.lower() == 'y':
            shutil.rmtree(CHROMA_PATH)
            print("Clearing and re-creating vector database.")
        else:
            print("Using existing vector DB. Exiting early.")
            return
    
    raw_documents = load_documents(PDF_PATH)
    chunks = chunk_documents(800, 200, raw_documents, 256)
    chunk_texts = [chunk['text'] for chunk in chunks]
    chunk_metadatas = [chunk['metadata'] for chunk in chunks]
    ids = [f"ID{i}" for i in range(len(chunk_texts))]
    create_and_update_vector_db(EMBED_MODEL_ID, chunk_texts, chunk_metadatas, ids)


# --- STEP 1: Load PDF ---
def load_documents(path):

    loader = PyPDFLoader(path)
    raw_documents = loader.load()

    return raw_documents

# raw_documents = load_documents(PDF_PATH)


# --- STEP 2: Chunking ---
def chunk_documents(chunk_size, chunk_overlap, raw_documents, max_tokens):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(raw_documents)

    formatted_chunk_list = [{"text": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]

    chunker = SemanticChunker(max_tokens=max_tokens)
    merged_chunks = chunker.chunk(formatted_chunk_list)

    simplified_chunks = []
    for chunk in merged_chunks:
        simplified_chunks.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"][0]["metadata"]
        })
    # Each merged chunk will now have: {"text": ..., "metadata": [...]}
    return simplified_chunks


# def make_chunk_ids(chunks, metadatas):

#     ids = []
#     last_page_id = None
#     current_chunk_index = 0

#     for i, chunk in enumerate(chunks):
#         filename = str(chunk.meta.origin.filename)
#         title = str(chunk.meta.headings[0] if chunk.meta.headings else None)

#         page_numbers = sorted(set(
#         prov.page_no
#         for item in chunk.meta.doc_items
#         for prov in item.prov
#         ))
#         current_page_numbers = ",".join(map(str, page_numbers)) if page_numbers else None
#         current_page_id = f"{filename}:{title}:{current_page_numbers}"
#         # If the page ID is the same as the last one, increment the index.
#         if current_page_id == last_page_id:
#             current_chunk_index += 1
#         else:
#             current_chunk_index = 0
#         # Calculate the chunk ID.
#         chunk_id = f"{current_page_id}:{current_chunk_index}"
#         last_page_id = current_page_id
#         ids.append(chunk_id)

#         metadatas[i]["id"] = chunk_id
    
#     return ids


# --- STEP 3: Embedding + Vector DB Setup ---
def create_and_update_vector_db(embedding_model, chunk_texts, metadatas, ids):
    
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
    collection.upsert(documents=chunk_texts, metadatas=metadatas, ids = ids)


if __name__ == "__main__":
    main()