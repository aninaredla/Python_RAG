from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from semantic_chunker.core import SemanticChunker
import time
import concurrent.futures

# Setup
PDF_PATH = "test_pdfs"
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
    
    
    # raw_documents = load_documents(PDF_PATH)
    raw_documents = load_documents_from_directory(PDF_PATH)
    # print(len(raw_documents))

    chunks = chunk_documents(800, 200, raw_documents, 256)
    chunk_texts = [chunk['text'] for chunk in chunks]
    chunk_metadatas = [chunk['metadata'] for chunk in chunks]
    ids = make_chunk_ids(chunks)

    create_and_update_vector_db(EMBED_MODEL_ID, chunk_texts, chunk_metadatas, ids)


def load_documents_from_directory(directory):

    def load_document(file_path):
        loader = PyPDFLoader(file_path)
        file_docs = loader.load()
        print(f"Successfully loaded {os.path.basename(file_path)}")
        return file_docs

    try:
        filenames = os.listdir(directory)
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
        return []
    except PermissionError:
        print(f"Permission denied for directory: {directory}")
        return []

    file_paths = [os.path.join(directory, f) for f in filenames if f.endswith(".pdf")]

    start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(load_document, file_paths)

    raw_docs = []
    for doc_list in results:
        raw_docs.extend(doc_list)

    finish = time.perf_counter()
    print(f"Process finished in {finish-start:.2f} seconds.")

    return raw_docs


# --- STEP 2: Chunking ---
def chunk_documents(chunk_size, chunk_overlap, raw_documents_batch, max_tokens):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(raw_documents_batch)

    formatted_chunk_list = [{"text": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]

    chunker = SemanticChunker(max_tokens=max_tokens)
    merged_chunks = chunker.chunk(formatted_chunk_list)

    simplified_chunks = []
    for chunk in merged_chunks:
        simplified_chunks.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"][0]["metadata"]
        })

    return simplified_chunks


def make_chunk_ids(chunks):
    ids = []
    page_counters = defaultdict(int)

    for chunk in chunks:
        metadata = chunk["metadata"]

        filename = os.path.splitext(os.path.basename(str(metadata.get("source", "unknown"))))[0].lower()
        title = str(metadata.get("title", "None")).strip() or "None"
        page_label = metadata.get("page_label", metadata.get("page", "None"))
        page_label = f"page_{page_label}"

        key = f"{filename}:{title}:{page_label}"
        chunk_index = page_counters[key]

        chunk_id = f"{key}:{chunk_index}"
        page_counters[key] += 1

        chunk["metadata"]["id"] = chunk_id
        ids.append(chunk_id)

    return ids


# --- STEP 3: Embedding + Vector DB Setup ---
def create_and_update_vector_db(embedding_model, chunk_texts, metadatas, ids):
    
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
    collection.upsert(documents=chunk_texts, metadatas=metadatas, ids = ids)


if __name__ == "__main__":
    main()