-- Migration 005 — Ajout colonne features à entity_relations
-- Stocke les features ML (name_sim, value_overlap, topo_distance, etc.)
-- pour l'explication du score (Item 51 CDC)

-- Ajout colonne features (JSONB) si elle n'existe pas
ALTER TABLE entity_relations
    ADD COLUMN IF NOT EXISTS features JSONB DEFAULT '{}';

-- Index GIN pour recherche rapide dans les features
CREATE INDEX IF NOT EXISTS idx_entity_relations_features
    ON entity_relations USING gin(features);

-- Vérification
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'entity_relations'
  AND column_name = 'features';