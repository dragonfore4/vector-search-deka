# search.py
import re

from chromadb import HttpClient
from ollama import Client
from pythainlp.tokenize import word_tokenize

# -------- CONFIGURATION --------
OLLAMA_MODEL = "bge-m3"
OLLAMA_HOST = "http://localhost:11434"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
CHROMA_COLLECTION = "pdf-index"
# -------------------------------


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_query_terms(query_text):
    return tokenize_terms(query_text)


def tokenize_terms(text):
    text = normalize_text(text)
    if not text:
        return []

    # Keep Thai-aware segmentation and also support latin/number terms.
    tokens = word_tokenize(text, keep_whitespace=False)
    cleaned = []
    for token in tokens:
        token = re.sub(r"[^0-9a-zA-Zก-๙]", "", token)
        if len(token) > 1:
            cleaned.append(token)

    return cleaned


def extract_character_ngrams(text, n=4):
    text = normalize_text(text).replace(" ", "")
    if len(text) < n:
        return []
    return [text[index : index + n] for index in range(len(text) - n + 1)]


def lexical_score(query_text, document_text):
    query_norm = normalize_text(query_text)
    document_norm = normalize_text(document_text)

    if not query_norm or not document_norm:
        return 0.0

    score = 0.0

    if query_norm in document_norm:
        score += 2.0

    query_terms = extract_query_terms(query_text)
    if query_terms:
        document_terms = set(tokenize_terms(document_text))
        if document_terms:
            matched_terms = sum(1 for term in query_terms if term in document_terms)
            score += matched_terms / len(query_terms)

    query_ngrams = extract_character_ngrams(query_text)
    if query_ngrams:
        matched_ngrams = sum(1 for ngram in query_ngrams if ngram in document_norm)
        score += 0.5 * (matched_ngrams / len(query_ngrams))

    return score


def combine_scores(distance, lexical):
    vector_score = 1.0 / (1.0 + distance)
    return vector_score + lexical


def search_similar_documents(query_text, n_results=3):
    ollama_client = Client(host=OLLAMA_HOST)
    chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    try:
        collection = chroma_client.get_collection(name=CHROMA_COLLECTION)
    except Exception as e:
        print(f"[ERROR] Could not connect to ChromaDB collection. Have you run ingest.py yet? ({e})")
        return

    print(f"\n[INFO] Generating embedding for your context: '{query_text}'...")

    response = ollama_client.embeddings(model=OLLAMA_MODEL, prompt=query_text)
    query_embedding = response["embedding"]

    print("[INFO] Searching ChromaDB for similar documents...\n")
    fetch_results = max(n_results * 5, 10)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_results,
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        print("No matches found.")
        return

    ranked_results = []
    for index, doc_text in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][index]
        distance = results["distances"][0][index] if "distances" in results and results["distances"] else 1.0
        lexical = lexical_score(query_text, doc_text)
        combined = combine_scores(distance, lexical)
        ranked_results.append(
            {
                "document": doc_text,
                "metadata": metadata,
                "distance": distance,
                "lexical": lexical,
                "combined": combined,
            }
        )

    ranked_results.sort(key=lambda item: item["combined"], reverse=True)
    ranked_results = ranked_results[:n_results]

    print("=== SEARCH RESULTS ===")
    for index, result in enumerate(ranked_results):
        metadata = result["metadata"]
        filename = metadata.get("source", "Unknown")
        page = metadata.get("page", "Unknown")
        chunk = metadata.get("chunk", "Unknown")

        print(f"Result {index + 1}:")
        print(f"📄 Document: {filename} (Page {page})")
        print(f"🧩 Chunk: {chunk}")
        print(f"🎯 Distance Score: {result['distance']:.4f} (Lower is more similar)")
        print(f"🔎 Lexical Boost: {result['lexical']:.4f}")
        print(f"⭐ Combined Score: {result['combined']:.4f}")
        print(f"📝 Matched Text:\n{result['document']}\n")
        print("-" * 50)


if __name__ == "__main__":
    print("==================================================")
    print(" PDF Semantic Search (พิมพ์ 'exit' หรือ 'quit' เพื่อออก)")
    print("==================================================")

    while True:
        user_input = input("\n🔍 พิมพ์คำถามหรือสิ่งที่คุณต้องการค้นหา: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("👋 ปิดโปรแกรม...")
            break

        if not user_input:
            continue

        search_similar_documents(user_input, n_results=3)