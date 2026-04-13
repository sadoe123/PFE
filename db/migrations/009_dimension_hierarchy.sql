-- ============================================================
-- Migration 009 — Hiérarchies dimensions analytiques
-- OnePilot Layer 2 — §2.2.3.B — 08/04/2026
-- ============================================================

-- Colonne dimension_hierarchy sur entity_fields
-- Stocke les expressions SQL générées par build_field_hierarchy()
-- ex: {"type":"time","levels":[{"level":"year","expr":"YEAR(ORDER_DATE)"},...]}
ALTER TABLE entity_fields
    ADD COLUMN IF NOT EXISTS dimension_hierarchy JSONB DEFAULT NULL;

-- Index GIN pour requêtes rapides sur les hiérarchies
CREATE INDEX IF NOT EXISTS idx_entity_fields_dim_hierarchy
    ON entity_fields USING GIN (dimension_hierarchy)
    WHERE dimension_hierarchy IS NOT NULL;

-- Index sur dimension_type (si pas déjà créé dans 008)
CREATE INDEX IF NOT EXISTS idx_entity_fields_dim_type
    ON entity_fields (dimension_type)
    WHERE dimension_type IS NOT NULL;

-- ============================================================
-- Vérification
-- SELECT column_name FROM information_schema.columns
-- WHERE table_name = 'entity_fields'
--   AND column_name = 'dimension_hierarchy';
-- ============================================================