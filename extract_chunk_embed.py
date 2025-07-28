from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
import shutil
# from pdfplumber_functions import extract_tables_with_numbering

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
    
    doc = create_docling_doc(PDF_PATH)
    chunks, chunk_texts = get_chunks_and_chunk_texts(EMBED_MODEL_ID, MAX_TOKENS, TOKEN_OVERLAP, doc)
    metadatas = extract_metadatas(chunks)
    ids = make_chunk_ids(chunks, metadatas)
    create_and_update_vector_db(EMBED_MODEL_ID, chunk_texts, metadatas, ids)


# --- STEP 1: Load PDF ---
def create_docling_doc(data_path):

    converter = DocumentConverter()
    result = converter.convert(data_path)
    doc = result.document
    return doc


# --- STEP 2: Chunking ---
def get_chunks_and_chunk_texts(embedding_model, max_tokens, overlap_tokens, docling_doc):

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(embedding_model),
        max_tokens=max_tokens,
    )
        # Part 1: chunk full text
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True, overlap_tokens = overlap_tokens)
    chunk_iter = chunker.chunk(dl_doc=docling_doc)
    chunks = list(chunk_iter)
    for chunk in chunks:
        chunk.text = chunker.contextualize(chunk=chunk)
        # print(chunk.text)
    chunk_texts = [chunk.text for chunk in chunks]
    return chunks, chunk_texts

    # Part 2: chunk structured tables
# tables, table_metadatas = extract_tables_with_numbering(PDF_PATH)
# chunk_texts = [chunk.text for chunk in chunks]
# chunk_texts.extend(tables)

def extract_metadatas(chunks):
    def extract_metadata(chunk):

        page_numbers = sorted(set(
            prov.page_no
            for item in chunk.meta.doc_items
            for prov in item.prov
        ))

        return {
            "filename": chunk.meta.origin.filename,
            "title": chunk.meta.headings[0] if chunk.meta.headings else None,
            "page_numbers": ",".join(map(str, page_numbers)) if page_numbers else None
        }

    return [extract_metadata(chunk) for chunk in chunks]


def make_chunk_ids(chunks, metadatas):

    ids = []
    last_page_id = None
    current_chunk_index = 0

    for i, chunk in enumerate(chunks):
        filename = str(chunk.meta.origin.filename)
        title = str(chunk.meta.headings[0] if chunk.meta.headings else None)

        page_numbers = sorted(set(
        prov.page_no
        for item in chunk.meta.doc_items
        for prov in item.prov
        ))
        current_page_numbers = ",".join(map(str, page_numbers)) if page_numbers else None
        current_page_id = f"{filename}:{title}:{current_page_numbers}"
        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        ids.append(chunk_id)

        metadatas[i]["id"] = chunk_id
    
    return ids


# --- STEP 3: Embedding + Vector DB Setup ---
def create_and_update_vector_db(embedding_model, chunk_texts, metadatas, ids):
    
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
    collection.upsert(documents=chunk_texts, metadatas=metadatas, ids = ids)


if __name__ == "__main__":
    main()