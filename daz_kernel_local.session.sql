-- 0) Voraussetzung (falls nicht schon aktiv)
-- CREATE EXTENSION IF NOT EXISTS pgcrypto;
-- CREATE EXTENSION IF NOT EXISTS vector;   -- nur wenn du pgvector nutzt
CREATE TABLE IF NOT EXISTS cefr_source (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    code text NOT NULL UNIQUE,
    -- z.B. "cefr_2001", "cefr_cv_2020"
    title text NOT NULL,
    -- "CEFR (2001)" / "Companion Volume (2020)"
    publisher text,
    -- Council of Europe
    year int,
    language text NOT NULL DEFAULT 'de',
    -- Textsprache des Deskriptors (de/en/...)
    uri text,
    -- Link/Referenz
    meta jsonb NOT NULL DEFAULT '{}'::jsonb,
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS cefr_scale (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id uuid NOT NULL REFERENCES cefr_source(id) ON DELETE CASCADE,
    code text NOT NULL,
    -- z.B. "listening_overall", "interaction_conversation"
    title text NOT NULL,
    -- Anzeigename
    skill text NOT NULL,
    -- "listening|reading|spoken_interaction|spoken_production|writing|mediation"
    domain text,
    -- optional: public/personal/educational/occupational
    parent_id uuid REFERENCES cefr_scale(id),
    sort_order int NOT NULL DEFAULT 0,
    meta jsonb NOT NULL DEFAULT '{}'::jsonb,
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source_id, code)
);
CREATE TABLE IF NOT EXISTS cefr_descriptor (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id uuid NOT NULL REFERENCES cefr_source(id) ON DELETE CASCADE,
    scale_id uuid REFERENCES cefr_scale(id) ON DELETE
    SET NULL,
        level text NOT NULL,
        -- A1..C2 (ggf. A0 später)
        descriptor text NOT NULL,
        -- der eigentliche "can do"-Satz
        descriptor_short text,
        -- optional kurze UI-Variante
        context_notes text,
        -- optional (Bedingungen/Beispiele)
        is_negative boolean NOT NULL DEFAULT false,
        -- falls du mal "cannot" Varianten importierst
        sort_order int NOT NULL DEFAULT 0,
        meta jsonb NOT NULL DEFAULT '{}'::jsonb,
        is_active boolean NOT NULL DEFAULT true,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now()
);
-- M:N zu deinem bestehenden Tag-System (topic/subtopic/grammar/rag_term/method…)
CREATE TABLE IF NOT EXISTS cefr_descriptor_tags (
    descriptor_id uuid NOT NULL REFERENCES cefr_descriptor(id) ON DELETE CASCADE,
    tag_id uuid NOT NULL REFERENCES p_tags(id) ON DELETE CASCADE,
    weight real NOT NULL DEFAULT 1.0,
    meta jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (descriptor_id, tag_id)
);
-- Optional: Embeddings (nur wenn du pgvector hast)
-- CREATE TABLE IF NOT EXISTS cefr_descriptor_embedding (
--   descriptor_id uuid PRIMARY KEY REFERENCES cefr_descriptor(id) ON DELETE CASCADE,
--   model text NOT NULL,                      -- z.B. "nomic-embed-text" / "text-embedding-3-large"
--   embedding vector(768) NOT NULL,           -- Dimension an dein Modell anpassen
--   created_at timestamptz NOT NULL DEFAULT now()
-- );
-- Indexe (wichtig)
CREATE INDEX IF NOT EXISTS ix_cefr_descriptor_level ON cefr_descriptor(level);
CREATE INDEX IF NOT EXISTS ix_cefr_descriptor_scale ON cefr_descriptor(scale_id);
CREATE INDEX IF NOT EXISTS ix_cefr_descriptor_source ON cefr_descriptor(source_id);
CREATE INDEX IF NOT EXISTS ix_cefr_scale_skill ON cefr_scale(skill);