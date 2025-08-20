from collections import defaultdict
import json
import math
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
import shutil
from langchain.document_loaders import PyPDFLoader
import concurrent.futures

# Setup
PDF_PATH = "test_pdfs"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "driving_manual_chunks"
CHUNKS_PATH = "chunks_dict_list.json"

def load_and_chunk_documents(directory, chunk_size=800, chunk_overlap=200):
    """Load PDFs and chunk them in one step."""
    try:
        file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pdf")]
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error accessing directory {directory}: {e}")
        return []
    
    def process_file(file_path):
        docs = PyPDFLoader(file_path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(docs)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_file, file_paths)
    
    chunks = []
    for chunk_list in results:
        chunks.extend(chunk_list)
    
    return chunks


def format_chunks_with_ids(chunks):
    """Format chunks and generate unique IDs."""
    page_counters = defaultdict(int)
    formatted_chunks = []
    
    for chunk in chunks:
        metadata = chunk.metadata
        filename = os.path.splitext(os.path.basename(str(metadata.get("source", "unknown"))))[0].lower()
        title = str(metadata.get("title", "None")).strip() or "None"
        page_label = f"page_{metadata.get('page_label', metadata.get('page', 'None'))}"
        
        key = f"{filename}:{title}:{page_label}"
        chunk_index = page_counters[key]
        chunk_id = f"{key}:{chunk_index}"
        page_counters[key] += 1
        
        formatted_chunks.append({
            'text': chunk.page_content,
            'metadata': {**metadata, 'id': chunk_id},
            'id': chunk_id
        })
    
    return formatted_chunks


def save_chunks_json(chunks, path):
    """Save chunks to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    return path


def create_vector_db(chunks_path, num_batches=3):
    """Create and populate ChromaDB with chunks."""
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_ID)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)

    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    batch_size = math.ceil(len(chunks) / num_batches)
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

    def upsert_batch(batch):
        collection.upsert(
            documents=[chunk['text'] for chunk in batch],
            metadatas=[chunk['metadata'] for chunk in batch],
            ids=[chunk['id'] for chunk in batch]
        )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(upsert_batch, batches)


def main():
    
    if os.path.exists(CHROMA_PATH):
        if input("Clear and re-create database (y/n)? : ").lower() == 'y':
            shutil.rmtree(CHROMA_PATH)
            if os.path.exists(CHUNKS_PATH):
                os.remove(CHUNKS_PATH)
            print("Clearing and re-creating vector database and chunks file.")
        else:
            print("Using existing vector DB. Exiting early.")
            return
    
    # Process pipeline
    chunks = load_and_chunk_documents(PDF_PATH)
    formatted_chunks = format_chunks_with_ids(chunks)
    save_chunks_json(formatted_chunks, CHUNKS_PATH)
    create_vector_db(CHUNKS_PATH)
    
    print(f"✅ Processed {len(formatted_chunks)} chunks into vector database.")


if __name__ == "__main__":
    main()