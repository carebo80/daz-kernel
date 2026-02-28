CREATE TABLE IF NOT EXISTS p_phase_models (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid (),
    code text UNIQUE NOT NULL, -- "rita", "aviva", "sos", "five_phases"
    title text NOT NULL, -- "RITA", "AVIVA", ...
    description text DEFAULT '',
    schema jsonb NOT NULL DEFAULT '{}'::jsonb, -- Modelldefinition (Phasen, Reihenfolge, Defaults)
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE p_units
ADD COLUMN IF NOT EXISTS phase_model_id uuid NULL REFERENCES p_phase_models (id);