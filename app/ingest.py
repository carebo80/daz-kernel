import os
import glob
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import psycopg2


@dataclass
class Chunk:
    source: str
    page: Optional[int]          # 1-basiert
    chunk_index: int
    content: str


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def extract_pdf_chunks(path: str, chunk_size: int, overlap: int) -> List[Chunk]:
    reader = PdfReader(path)
    out: List[Chunk] = []
    src = os.path.basename(path)
    cix = 0
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for piece in chunk_text(page_text, chunk_size, overlap):
            out.append(Chunk(source=src, page=i, chunk_index=cix, content=piece))
            cix += 1
    return out


def upsert_chunks(conn, chunks: List[Chunk], embeddings: List[List[float]]):
    # INSERT mit ON CONFLICT für uq_doc_chunks_source_page_chunk
    sql = """
    INSERT INTO doc_chunks (source, page, chunk_index, content, embedding)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (source, page, chunk_index)
    DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding;
    """
    with conn.cursor() as cur:
        for ch, emb in zip(chunks, embeddings):
            # pgvector erwartet Stringformat: '[0.1, 0.2, ...]'
            emb_str = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"
            cur.execute(sql, (ch.source, ch.page, ch.chunk_index, ch.content, emb_str))
    conn.commit()


def main():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    model_name = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size = int(os.getenv("CHUNK_SIZE", "900"))
    overlap = int(os.getenv("CHUNK_OVERLAP", "150"))

    if not db_url:
        raise RuntimeError("DATABASE_URL fehlt in .env")

    # Optional: Pfad per CLI, sonst default ./docs
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.getcwd(), "docs")
    docs_dir = os.path.abspath(docs_dir)

    pdfs = sorted(glob.glob(os.path.join(docs_dir, "**/*.pdf"), recursive=True))
    if not pdfs:
        print(f"Keine PDFs in {docs_dir} gefunden.")
        return

    print(f"Embedding Model: {model_name}")
    embedder = SentenceTransformer(model_name)

    conn = psycopg2.connect(db_url)

    total_chunks = 0
    for pdf in pdfs:
        print(f"\n📄 Ingest: {pdf}")
        chunks = extract_pdf_chunks(pdf, chunk_size, overlap)
        if not chunks:
            print("  -> keine Text-Chunks extrahiert (evtl. Scan-PDF).")
            continue

        texts = [c.content for c in chunks]
        embs = embedder.encode(texts, normalize_embeddings=True).tolist()
        upsert_chunks(conn, chunks, embs)

        total_chunks += len(chunks)
        print(f"  -> {len(chunks)} Chunks gespeichert.")

    conn.close()
    print(f"\n✅ Fertig. Total Chunks gespeichert: {total_chunks}")


if __name__ == "__main__":
    main()
