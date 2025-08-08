from collections import defaultdict
import json
import math
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
import shutil
from langchain.document_loaders import PyPDFLoader
from semantic_chunker.core import SemanticChunker
import concurrent.futures

# # Setup
PDF_PATH = "test_pdfs"
MAX_TOKENS = 256
TOKEN_OVERLAP = 64
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "docling_chunks"


def load_documents_from_directory(directory):

    def load_document(file_path):
        loader = PyPDFLoader(file_path)
        file_docs = loader.load()
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

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(load_document, file_paths)

    raw_docs = []
    for doc_list in results:
        raw_docs.extend(doc_list)

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


def format_chunks(all_chunks):

    documents = [chunk["text"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]
    ids = make_chunk_ids(all_chunks)

    chunks_dict_list = []
    for i in range(len(all_chunks)):
        chunks_dict_list.append({
            'text': documents[i],
            'metadata': metadatas[i],
            'id': ids[i]
        })
    
    return chunks_dict_list


def chunks_to_json(chunks_dict_list):

    chunks_path = 'chunks_dict_list.json'
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_dict_list, f, ensure_ascii=False, indent=4)
    
    return chunks_path


# --- STEP 3: Embedding + Vector DB Setup ---
def create_and_update_vector_db(chunks_path, num_batches = 3):
    
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_ID)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)

    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks_dict_list = json.load(f)

    batch_size = math.ceil(len(chunks_dict_list)/num_batches)
    batches = [chunks_dict_list[i:i + batch_size] for i in range(0, len(chunks_dict_list), batch_size)]

    def upsert_individual_batch(chunk_batch):
        documents = [chunk['text'] for chunk in chunk_batch]
        metadatas = [chunk['metadata'] for chunk in chunk_batch]
        ids = [chunk['id'] for chunk in chunk_batch]
        collection.upsert(documents=documents, metadatas=metadatas, ids = ids)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(upsert_individual_batch, batches)


def main():

    if os.path.exists(CHROMA_PATH):
        clear_db = input("Clear and re-create database (y/n)? :  ")
        if clear_db.lower() == 'y':
            shutil.rmtree(CHROMA_PATH)
            print("Clearing and re-creating vector database.")
        else:
            print("Using existing vector DB. Exiting early.")
            return
    
    raw_documents = load_documents_from_directory(PDF_PATH)

    chunks = chunk_documents(800, 200, raw_documents, 256)
    chunks_dict_list = format_chunks(chunks)
    chunks_path = chunks_to_json(chunks_dict_list)

    create_and_update_vector_db(chunks_path)


if __name__ == "__main__":
    main()