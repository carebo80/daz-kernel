INSERT INTO
    p_phase_models (
        code,
        title,
        description,
        schema
    )
VALUES (
        'five_phases',
        '5-Phasen-Modell',
        'Einführung/Vorbereitung → Präsentation → Bewusstmachung → Festigung → Transfer',
        '{
    "phases": [
      {"key":"einführung", "title":"Einführung/Vorbereitung"},
      {"key":"präsentation", "title":"Präsentation"},
      {"key":"bewusstmachung", "title":"Bewusstmachung"},
      {"key":"festigung", "title":"Festigung"},
      {"key":"transfer", "title":"Transfer"}
    ]
  }'::jsonb
    )
ON CONFLICT (code) DO NOTHING;

INSERT INTO
    p_phase_models (
        code,
        title,
        description,
        schema
    )
VALUES (
        'sos_grammatik',
        'SOS Grammatik',
        'Sammeln → Ordnen → Systematisieren',
        '{
    "phases": [
      {"key":"sammeln", "title":"Sammeln"},
      {"key":"ordnen", "title":"Ordnen"},
      {"key":"systematisieren", "title":"Systematisieren"}
    ]
  }'::jsonb
    )
ON CONFLICT (code) DO NOTHING;