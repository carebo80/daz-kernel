-- db/schema.sql
-- Minimaler, zukunftsfähiger Kern: Dokument-Chunks + Zitierfähige Outputs

BEGIN;

-- 1) pgvector aktivieren
CREATE EXTENSION IF NOT EXISTS vector;

-- 2) Dokument-Chunks (RAG-Wissensbasis)
CREATE TABLE IF NOT EXISTS doc_chunks (
  id          BIGSERIAL PRIMARY KEY,
  source      TEXT NOT NULL,        -- Dateiname oder Pfad, z.B. "GER_Tabelle1_Globalskala.pdf"
  page        INT,                  -- 1-basiert, wenn du willst
  chunk_index INT NOT NULL,
  content     TEXT NOT NULL,
  embedding   vector(384),          -- passt zu all-MiniLM-L6-v2 (384)
  created_at  TIMESTAMPTZ DEFAULT now()
);

-- Optional: Duplikate vermeiden (gleiche Quelle/Seite/Chunk)
CREATE UNIQUE INDEX IF NOT EXISTS uq_doc_chunks_source_page_chunk
ON doc_chunks (source, page, chunk_index);

-- 3) Vektor-Index (schneller Similarity Search)
-- Hinweis: ivfflat ist gut; braucht später ANALYZE für beste Performance
CREATE INDEX IF NOT EXISTS ix_doc_chunks_embedding_ivfflat
ON doc_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- 4) Generierte Outputs (z.B. Feinplanung, AB, Feedback) + Quellen
CREATE TABLE IF NOT EXISTS generated_outputs (
  id         BIGSERIAL PRIMARY KEY,
  kind       TEXT NOT NULL,         -- "lesson_plan" | "worksheet" | "feedback" | ...
  input      JSONB NOT NULL,        -- Parameter: Niveau, Thema, Dauer, etc.
  prompt     TEXT NOT NULL,
  output     TEXT NOT NULL,
  citations  JSONB,                 -- [{source,page,chunk_index,score}, ...]
  created_at TIMESTAMPTZ DEFAULT now()
);

-- 5) (Optional) Skills/Attempts fürs Game (kannst du später nutzen)
CREATE TABLE IF NOT EXISTS users (
  id         BIGSERIAL PRIMARY KEY,
  username   TEXT UNIQUE NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS skills (
  id          BIGSERIAL PRIMARY KEY,
  user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  competence  TEXT NOT NULL,         -- z.B. "A2_INTERAKTION"
  level       DOUBLE PRECISION NOT NULL DEFAULT 0.0,  -- 0..1
  updated_at  TIMESTAMPTZ DEFAULT now(),
  UNIQUE(user_id, competence)
);

CREATE TABLE IF NOT EXISTS attempts (
  id          BIGSERIAL PRIMARY KEY,
  user_id     BIGINT REFERENCES users(id) ON DELETE SET NULL,
  competence  TEXT,
  task_json   JSONB,
  answer_text TEXT,
  score       DOUBLE PRECISION,
  feedback    TEXT,
  citations   JSONB,
  created_at  TIMESTAMPTZ DEFAULT now()
);

COMMIT;
