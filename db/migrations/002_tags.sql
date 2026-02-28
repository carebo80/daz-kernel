-- db/migrations/002_tags.sql
-- Tags/Taxonomie (Topics, Subtopics, Grammar, Methods, Levels, RAG-Terms, …)
-- + Mapping Unit <-> Tags

-- Voraussetzung (hast du schon, aber sicher ist sicher):
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS p_tags (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid (),
    type text NOT NULL, -- z.B. topic, subtopic, grammar, method, level, rag_term, skill
    code text NOT NULL, -- slug-like, eindeutig je type
    title text NOT NULL, -- Anzeige
    parent_id uuid NULL REFERENCES p_tags (id) ON DELETE SET NULL, -- Hierarchie
    is_active boolean NOT NULL DEFAULT true,
    meta jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (type, code)
);

CREATE INDEX IF NOT EXISTS ix_p_tags_type ON p_tags(type);

CREATE INDEX IF NOT EXISTS ix_p_tags_parent ON p_tags (parent_id);

-- Mapping: Units <-> Tags (optional "role": primary/secondary/etc.)
CREATE TABLE IF NOT EXISTS p_unit_tags (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid (),
    unit_id uuid NOT NULL REFERENCES p_units (id) ON DELETE CASCADE,
    tag_id uuid NOT NULL REFERENCES p_tags (id) ON DELETE CASCADE,
    role text NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (unit_id, tag_id)
);

CREATE INDEX IF NOT EXISTS ix_p_unit_tags_unit ON p_unit_tags (unit_id);

CREATE INDEX IF NOT EXISTS ix_p_unit_tags_tag ON p_unit_tags (tag_id);