CREATE TABLE IF NOT EXISTS p_assets (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  type text NOT NULL,
  -- 'image' | 'audio'
  uri text NOT NULL,
  -- filepath oder s3 url oder /static/...
  mime text,
  width int,
  height int,
  sha256 text,
  source text,
  -- 'comfyui' | 'manual' | 'stock' ...
  prompt text,
  model text,
  meta jsonb NOT NULL DEFAULT '{}'::jsonb,
  is_active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_assets_type ON p_assets(type);
CREATE INDEX IF NOT EXISTS ix_assets_sha ON p_assets(sha256);
CREATE TABLE IF NOT EXISTS p_vocabulary_assets (
  vocabulary_id uuid NOT NULL REFERENCES p_vocabulary(id) ON DELETE CASCADE,
  asset_id uuid NOT NULL REFERENCES p_assets(id) ON DELETE CASCADE,
  role text NOT NULL DEFAULT 'main',
  -- main|alt|icon|story|worksheet
  sort_order int NOT NULL DEFAULT 0,
  PRIMARY KEY (vocabulary_id, asset_id)
);
CREATE INDEX IF NOT EXISTS ix_vocab_assets_vocab ON p_vocabulary_assets(vocabulary_id);