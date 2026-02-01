BEGIN;

-- UUIDs
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Levels (A1..C2)
CREATE TABLE IF NOT EXISTS p_levels (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  code TEXT NOT NULL UNIQUE,     -- "A1", "A2", ...
  title TEXT NOT NULL,           -- "Grundlagen" etc.
  constraints JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Topics (Bank, Arzt, Schule...)
CREATE TABLE IF NOT EXISTS p_topics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  slug TEXT NOT NULL UNIQUE,     -- "bank"
  title TEXT NOT NULL            -- "Bank"
);

-- Skills (dein Raster; später fürs Game)
CREATE TABLE IF NOT EXISTS p_skills (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  code TEXT NOT NULL UNIQUE,     -- z.B. "A2_BANK_OB_QUESTIONS"
  title TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  tags TEXT[] NOT NULL DEFAULT '{}'
);

-- Units (eine geplante Einheit inkl. Plan + Sprachmittel)
CREATE TABLE IF NOT EXISTS p_units (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  level_id UUID NOT NULL REFERENCES p_levels(id),
  topic_id UUID REFERENCES p_topics(id),

  time_start TEXT NOT NULL DEFAULT '',
  time_end   TEXT NOT NULL DEFAULT '',
  strong_group BOOLEAN NOT NULL DEFAULT false,

  title TEXT NOT NULL DEFAULT '',
  notes TEXT NOT NULL DEFAULT '',

  -- Feinplanung als JSON (Phasen, Aufgaben, Material, Sozialform etc.)
  plan JSONB NOT NULL DEFAULT '{}'::jsonb,

  -- Sprachmittel als JSON (vocabulary/phrases/grammar_focus/mini_dialogues)
  language_support JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_p_units_level ON p_units(level_id);
CREATE INDEX IF NOT EXISTS idx_p_units_topic ON p_units(topic_id);

-- Unit <-> Skills (many-to-many)
CREATE TABLE IF NOT EXISTS p_unit_skills (
  unit_id UUID NOT NULL REFERENCES p_units(id) ON DELETE CASCADE,
  skill_id UUID NOT NULL REFERENCES p_skills(id),
  PRIMARY KEY (unit_id, skill_id)
);

-- Citations auf doc_chunks (Beleg-Layer aus deinem RAG-Korpus)
CREATE TABLE IF NOT EXISTS p_unit_citations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  unit_id UUID NOT NULL REFERENCES p_units(id) ON DELETE CASCADE,
  chunk_id BIGINT NOT NULL REFERENCES doc_chunks(id),
  score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  quote TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_p_unit_citations_unit ON p_unit_citations(unit_id);
CREATE INDEX IF NOT EXISTS idx_p_unit_citations_chunk ON p_unit_citations(chunk_id);

COMMIT;
