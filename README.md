# Vector DB PDF Search

This project ingests PDF files into a ChromaDB collection and lets you search them with semantic embeddings from Ollama.

## What It Does

- `ingest.py` reads all PDF files from `./2568-2569/`
- Each PDF is split into smaller text chunks
- The chunks are embedded with Ollama using `bge-m3`
- The embeddings are stored in ChromaDB under the collection `multi_pdf_vectors`
- `search.py` lets you type a query and retrieve the most similar chunks

## Requirements

Install these prerequisites before running the scripts:

- Python 3.10 or newer
- Docker
- Ollama installed locally and running on `http://localhost:11434`
- The Ollama embedding model `bge-m3`

Python packages used by the scripts:

- `chromadb`
- `ollama`
- `PyMuPDF`
- `langchain-text-splitters`

## Setup

1. Start ChromaDB with Docker:

   ```bash
   docker compose up -d chromadb
   ```

2. Start Ollama:

   ```bash
   ollama serve
   ```

3. Pull the embedding model used by the project:

   ```bash
   ollama pull bge-m3
   ```

4. Install the Python dependencies:

   ```bash
   pip install chromadb ollama PyMuPDF langchain-text-splitters
   ```

## Prepare Your PDFs

Put the PDF files you want to search into the `2568-2569/` folder.

If the folder does not exist, `ingest.py` will create it and stop so you can add files.

## Ingest Documents

Run the ingestion script to build the vector database:

```bash
python ingest.py
```

Important behavior:

- The script deletes the existing `multi_pdf_vectors` collection before ingesting new data
- PDFs are split into chunks of about 1000 characters with 200 characters of overlap
- Each chunk stores the source filename and page number as metadata

## Search Documents

After ingestion, start the interactive search script:

```bash
python search.py
```

Then type a question or keyword query. Example:

```text
สรุปหัวข้อความเสี่ยงด้านเครดิต
```

Type `exit` or `quit` to close the search loop.

## Project Structure

- `compose.yaml` - Docker service definition for ChromaDB
- `ingest.py` - PDF ingestion and embedding pipeline
- `search.py` - interactive semantic search script
- `2568-2569/` - input folder for PDF files
- `chroma_data/` - persistent ChromaDB storage

## Troubleshooting

- If ingestion fails, confirm that Ollama is running and `bge-m3` is installed
- If search says the collection is missing, run `ingest.py` first
- If Docker is not reachable, verify that the ChromaDB container is up on port `8000`

## Notes

- The project currently connects to Ollama at `http://localhost:11434`
- The project currently connects to ChromaDB at `http://localhost:8000`
- You can change the model, collection name, or chunking settings directly in `ingest.py` and `search.py`