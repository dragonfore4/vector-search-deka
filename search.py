# search.py
from ollama import Client
from chromadb import HttpClient

# -------- CONFIGURATION --------
OLLAMA_MODEL = "bge-m3"
OLLAMA_HOST = "http://localhost:11434"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
CHROMA_COLLECTION = "multi_pdf_vectors"
# -------------------------------

def search_similar_documents(query_text, n_results=3):
    ollama_client = Client(host=OLLAMA_HOST)
    chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    
    try:
        collection = chroma_client.get_collection(name=CHROMA_COLLECTION)
    except Exception as e:
        print(f"[ERROR] Could not connect to ChromaDB collection. Have you run ingest.py yet? ({e})")
        return

    print(f"\n[INFO] Generating embedding for your context: '{query_text}'...")
    
    # 1. Embed the user query using the same model
    response = ollama_client.embeddings(model=OLLAMA_MODEL, prompt=query_text)
    query_embedding = response["embedding"]

    # 2. Query ChromaDB for the closest matches
    print("[INFO] Searching ChromaDB for similar documents...\n")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    # 3. Display the results
    if not results['documents'] or not results['documents'][0]:
        print("No matches found.")
        return

    print("=== SEARCH RESULTS ===")
    # print("results:", results)
    for i in range(len(results['documents'][0])):
        doc_text = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i] if 'distances' in results and results['distances'] else "N/A"
        
        filename = metadata.get('source', 'Unknown')
        page = metadata.get('page', 'Unknown')

        print(f"Result {i+1}:")
        print(f"📄 Document: {filename} (Page {page})")
        print(f"🎯 Distance Score: {distance:.4f} (Lower is more similar)")
        print(f"📝 Matched Text:\n{doc_text}\n")
        print("-" * 50)

if __name__ == "__main__":
    print("==================================================")
    print(" PDF Semantic Search (พิมพ์ 'exit' หรือ 'quit' เพื่อออก)")
    print("==================================================")
    
    while True:
        # 1. รอรับข้อความจากผู้ใช้
        user_input = input("\n🔍 พิมพ์คำถามหรือสิ่งที่คุณต้องการค้นหา: ").strip()
        
        # 2. เช็คว่าผู้ใช้ต้องการออกจากโปรแกรมหรือไม่
        if user_input.lower() in ['exit', 'quit']:
            print("👋 ปิดโปรแกรม...")
            break
            
        # 3. ถ้าผู้ใช้กด Enter เปล่าๆ ให้ข้ามไปรอรับค่าใหม่
        if not user_input:
            continue
            
        # 4. ส่งข้อความไปค้นหา (กำหนดค่า default ของผลลัพธ์เป็น 3)
        search_similar_documents(user_input, n_results=3)