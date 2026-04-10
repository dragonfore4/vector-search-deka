import os
import re

import fitz  # PyMuPDF
from chromadb import HttpClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import Client

# -------- CONFIGURATION --------
PDF_FOLDER = "./2568-2569"
OLLAMA_MODEL = "bge-m3"
OLLAMA_HOST = "http://localhost:11434"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
CHROMA_COLLECTION = "pdf-index"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
COLLECTION_METADATA = {"hnsw:space": "cosine"}
# -------------------------------


def normalize_text(text):
    text = text.replace("\x00", "")
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_document_chunks(doc, text_splitter, filename):
    chunks = []

    for page_index, page in enumerate(doc):
        text = normalize_text(page.get_text("text"))
        if not text:
            continue

        page_chunks = text_splitter.split_text(text)
        for chunk_index, chunk_text in enumerate(page_chunks):
            chunk_text = normalize_text(chunk_text)
            if len(chunk_text) < 30:
                continue

            chunks.append(
                {
                    "id": f"{filename}_p{page_index}_c{chunk_index}",
                    "text": chunk_text,
                    "metadata": {
                        "source": filename,
                        "page": page_index,
                        "chunk": chunk_index,
                    },
                }
            )

    return chunks


def main():
    ollama_client = Client(host=OLLAMA_HOST)
    chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    try:
        chroma_client.delete_collection(name=CHROMA_COLLECTION)
        print(f"[INFO] Deleted existing collection '{CHROMA_COLLECTION}'")
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata=COLLECTION_METADATA,
    )

    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"[INFO] Created folder {PDF_FOLDER}. Add PDF files there and run again.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    pdf_files = [filename for filename in os.listdir(PDF_FOLDER) if filename.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"[INFO] No PDF files found in {PDF_FOLDER}")
        return

    for filename in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, filename)
        print(f"\n[READING] {filename}")

        try:
            with fitz.open(pdf_path) as doc:
                chunks = build_document_chunks(doc, text_splitter, filename)

            if not chunks:
                print(f"[INFO] Skipped {filename} because no usable text was found")
                continue

            ids, embeddings, metadatas, documents = [], [], [], []
            seen_texts = set()

            for chunk in chunks:
                if chunk["text"] in seen_texts:
                    continue

                try:
                    response = ollama_client.embeddings(model=OLLAMA_MODEL, prompt=chunk["text"])
                    ids.append(chunk["id"])
                    embeddings.append(response["embedding"])
                    metadatas.append(chunk["metadata"])
                    documents.append(chunk["text"])
                    seen_texts.add(chunk["text"])
                except Exception as e:
                    print(f"  > [ERROR] Embedding failed for {chunk['id']}: {e}")

            if ids:
                collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
                print(f"[SUCCESS] Indexed {filename} ({len(ids)} chunks)")

        except Exception as e:
            print(f"[ERROR] Could not read {filename}: {e}")


if __name__ == "__main__":
    main()