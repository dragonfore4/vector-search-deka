# ingest.py
import os
import fitz  # PyMuPDF
from ollama import Client
from chromadb import HttpClient
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------- CONFIGURATION --------
PDF_FOLDER = "./2568-2569"  # โฟลเดอร์ที่เก็บไฟล์ PDF
# OLLAMA_MODEL = "mxbai-embed-large:335m"
OLLAMA_MODEL = "bge-m3"
OLLAMA_HOST = "http://localhost:11434"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
CHROMA_COLLECTION = "multi_pdf_vectors"
CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 200
# -------------------------------

def main():
    ollama_client = Client(host=OLLAMA_HOST)
    chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    # --- เพิ่ม 3 บรรทัดนี้เพื่อล้างข้อมูลเก่า ---
    try:
        chroma_client.delete_collection(name=CHROMA_COLLECTION)
        print(f"[INFO] ลบ Collection '{CHROMA_COLLECTION}' เดิมทิ้งแล้ว")
    except Exception:
        pass # ถ้าไม่มี Collection อยู่ก่อนแล้วก็ให้ข้ามไป

    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)

    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"[INFO] สร้างโฟลเดอร์ {PDF_FOLDER} แล้ว กรุณานำไฟล์ PDF ไปใส่ไว้")
        return

    # Initialize the semantic text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""] # Splits by paragraph, then sentence, then word
    )

    # ลูปอ่านไฟล์ PDF ทุกไฟล์ในโฟลเดอร์
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            print(f"\n[READING] กำลังอ่าน: {filename}")
            
            try:
                # Use PyMuPDF (fitz) for better text extraction
                doc = fitz.open(pdf_path)
                all_chunks = []
                
                for i, page in enumerate(doc):
                    text = page.get_text("text") # Extract clean text
                    if text.strip():
                        # หั่น Chunk แบบ Semantic (รักษาความหมาย)
                        chunks = text_splitter.split_text(text)
                        for j, chunk_text in enumerate(chunks):
                            chunk_id = f"{filename}_p{i}_c{j}"
                            all_chunks.append((chunk_id, chunk_text.strip()))

                # ส่งไปทำ Embedding และเก็บลง ChromaDB
                ids, embeddings, metadatas, documents = [], [], [], []
                for doc_id, text in all_chunks:
                    try:
                        response = ollama_client.embeddings(model=OLLAMA_MODEL, prompt=text)
                        ids.append(doc_id)
                        embeddings.append(response["embedding"])
                        # Extract page number accurately for metadata
                        page_num = doc_id.split('_p')[1].split('_c')[0]
                        metadatas.append({"source": filename, "page": page_num})
                        documents.append(text)
                    except Exception as e:
                        print(f"  > [ERROR] Embeddings failed for {doc_id}: {e}")

                if ids:
                    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
                    print(f"[SUCCESS] เก็บข้อมูลจาก {filename} เรียบร้อย ({len(ids)} chunks)")
                    
            except Exception as e:
                print(f"[ERROR] ไม่สามารถอ่านไฟล์ {filename} ได้: {e}")

if __name__ == "__main__":
    main()