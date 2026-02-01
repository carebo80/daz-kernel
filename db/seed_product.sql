BEGIN;

INSERT INTO p_levels (code, title) VALUES
  ('A1', 'Grundlagen'),
  ('A2', 'Grundlagen'),
  ('B1', 'Selbständige Sprachverwendung'),
  ('B2', 'Selbständige Sprachverwendung'),
  ('C1', 'Kompetente Sprachverwendung'),
  ('C2', 'Kompetente Sprachverwendung')
ON CONFLICT (code) DO NOTHING;

INSERT INTO p_topics (slug, title) VALUES
('aemter-behoerden','Ämter und Behörden'),
('arbeit-beruf','Arbeit und Beruf'),
('aus-weiterbildung','Aus- und Weiterbildung'),
('kinder','Betreuung und Erziehung von Kindern'),
('dienstleistungen-bank-versicherungen','Dienstleistungen / Banken / Versicherungen'),
('einkaufen','Einkaufen / Handel / Konsum'),
('essen-trinken','Essen und Trinken'),
('freizeit','Freizeit'),
('gesundheit','Gesundheit und Hygiene / menschlicher Körper'),
('medien','Medien und Mediennutzung'),
('mobilitaet-verkehr','Orte / Mobilität und Verkehr'),
('natur-umwelt','Natur und Umwelt'),
('person-soziale-kontakte','Zur Person / soziale Kontakte'),
('unterricht','Unterricht'),
('wohnen','Wohnen')
ON CONFLICT (slug) DO NOTHING;

INSERT INTO p_skills (code, title, description, tags) VALUES
  ('A2_BANK_OB_QUESTIONS',
   'Indirekte Ja/Nein-Fragen mit „ob“ (Bank)',
   'TN formulieren direkte Ja/Nein-Fragen als indirekte Fragen mit „ob“ in Banksituationen (z.B. „Darf ich fragen, ob …?“).',
   ARRAY['A2','Bank','Grammatik','Sprechen'])
ON CONFLICT (code) DO NOTHING;

COMMIT;
