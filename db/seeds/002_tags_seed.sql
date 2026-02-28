-- db/seeds/002_tags_seed.sql

-- Levels
INSERT INTO
    p_tags (type, code, title)
VALUES ('level', 'a1', 'A1'),
    ('level', 'a2', 'A2'),
    ('level', 'b1', 'B1'),
    ('level', 'b2', 'B2'),
    ('level', 'c1', 'C1'),
    ('level', 'c2', 'C2')
ON CONFLICT (type, code) DO NOTHING;

-- Topics (grob)
INSERT INTO
    p_tags (type, code, title)
VALUES ('topic', 'bank', 'Bank'),
    ('topic', 'arzt', 'Arzt'),
    ('topic', 'wohnen', 'Wohnen'),
    (
        'topic',
        'arbeit',
        'Arbeit und Beruf'
    ),
    (
        'topic',
        'gesundheit_hygiene',
        'Gesundheit und Hygiene'
    ),
    (
        'topic',
        'versicherung',
        'Versicherung'
    ),
    (
        'topic',
        'aemter',
        'Ämter und Behörden'
    ),
    (
        'topic',
        'ausbildung',
        'Aus- und Weiterbildung'
    ),
    (
        'topic',
        'arbkinder_betreuung',
        'Betreuung und Erziehung von Kindern'
    ),
    (
        'topic',
        'handel_konsum',
        'Handel und Konsum'
    ),
    (
        'topic',
        'freizeit',
        'Freizeit'
    ),
    (
        'topic',
        'essen_trinken',
        'Essen und Trinken'
    ),
    (
        'topic',
        'medien',
        'Medien und Mediennutzung'
    ),
    (
        'topic',
        'mobilitaet_verkehr',
        'Orte, Mobilität und Verkehr'
    ),
    (
        'topic',
        'natur_umwelt',
        'Natur und Umwelt'
    ),
    (
        'topic',
        'unterricht',
        'Unterricht'
    ),
    (
        'topic',
        'person_sozial',
        'Zur Person, soziale Kontakte'
    )
ON CONFLICT (type, code) DO NOTHING;

-- Subtopics (unter Bank / Arzt)
INSERT INTO
    p_tags (type, code, title, parent_id)
SELECT 'subtopic', 'konto', 'Konto', t.id
FROM p_tags t
WHERE
    t.type = 'topic'
    AND t.code = 'bank'
ON CONFLICT (type, code) DO NOTHING;

INSERT INTO
    p_tags (type, code, title, parent_id)
SELECT 'subtopic', 'kreditkarte', 'Kreditkarte', t.id
FROM p_tags t
WHERE
    t.type = 'topic'
    AND t.code = 'bank'
ON CONFLICT (type, code) DO NOTHING;

INSERT INTO
    p_tags (type, code, title, parent_id)
SELECT 'subtopic', 'symptome', 'Symptome', t.id
FROM p_tags t
WHERE
    t.type = 'topic'
    AND t.code = 'arzt'
ON CONFLICT (type, code) DO NOTHING;

-- Grammar (mit Beispielen in meta)
INSERT INTO
    p_tags (type, code, title, meta)
VALUES (
        'grammar',
        'artikel',
        'Artikel (der/die/das)',
        '{"examples":["die Kreditkarte","eine Kreditkarte","das Konto"]}'::jsonb
    ),
    (
        'grammar',
        'verbposition',
        'Verbposition im Satz',
        '{"examples":["Ich möchte eine Kreditkarte.","Können Sie mir helfen?"]}'::jsonb
    ),
    (
        'grammar',
        'bitte-fragen',
        'Höfliche Bitten / Fragen mit bitte',
        '{"examples":["Darf ich bitte Wasser?","Könnte ich bitte einen Termin bekommen?"]}'::jsonb
    )
ON CONFLICT (type, code) DO NOTHING;

-- Methods / Tools (didaktische Elemente)
INSERT INTO
    p_tags (type, code, title, meta)
VALUES (
        'method',
        'lueckentext',
        'Lückentext',
        '{}'::jsonb
    ),
    (
        'method',
        'wortkarten',
        'Wortkarten',
        '{}'::jsonb
    ),
    (
        'method',
        'rollenspiel',
        'Rollenspiel',
        '{}'::jsonb
    ),
    (
        'method',
        'chunks',
        'Chunks / Textbausteine',
        '{}'::jsonb
    )
ON CONFLICT (type, code) DO NOTHING;

-- RAG-Terms (deine “Text-Terms” als kontrollierte Liste)
INSERT INTO
    p_tags (type, code, title)
VALUES (
        'rag_term',
        'routinem',
        'routinem'
    ),
    (
        'rag_term',
        'kontaktgespr',
        'Kontaktgespr'
    ),
    (
        'rag_term',
        'austausch',
        'Austausch'
    ),
    (
        'rag_term',
        'fragen',
        'Fragen'
    )
ON CONFLICT (type, code) DO NOTHING;