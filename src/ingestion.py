import argparse
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from src.config import INDEX_NAME, PDF_PATH, PDF_URL
from src.vector_store import ensure_index, get_embeddings


def download_pdf(destination: Path = PDF_PATH) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(PDF_URL, headers={"User-Agent": "agentic-ai-rag/1.0"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as pdf_file:
        pdf_file.write(response.read())
    return destination


def run_ingestion(pdf_path: Path = PDF_PATH, index_name: str = INDEX_NAME) -> int:
    if not pdf_path.exists():
        download_pdf(pdf_path)

    pages = PyPDFLoader(str(pdf_path)).load()
    if not pages:
        raise ValueError(f"No pages were extracted from {pdf_path}.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        add_start_index=True,
    ).split_documents(pages)
    for chunk in chunks:
        chunk.metadata["source"] = pdf_path.name

    client = ensure_index(index_name)
    vector_store = PineconeVectorStore(
        index=client.Index(index_name),
        embedding=get_embeddings(),
    )
    ids = [
        hashlib.sha256(
            f"{chunk.metadata.get('page', 0)}:{chunk.page_content}".encode("utf-8")
        ).hexdigest()
        for chunk in chunks
    ]
    vector_store.add_documents(documents=chunks, ids=ids)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the Agentic AI ebook into Pinecone.")
    parser.add_argument("--pdf", type=Path, default=PDF_PATH, help="Path to the source PDF")
    parser.add_argument("--index", default=INDEX_NAME, help="Pinecone index name")
    args = parser.parse_args()
    count = run_ingestion(args.pdf, args.index)
    print(f"Indexed {count} chunks into Pinecone index '{args.index}'.")


if __name__ == "__main__":
    main()