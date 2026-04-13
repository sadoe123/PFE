-- ============================================================
-- Migration 007 — Default values sur les colonnes
-- OnePilot Layer 2 — Session 2 — 08/04/2026
-- ============================================================

ALTER TABLE entity_fields
    ADD COLUMN IF NOT EXISTS default_value TEXT;

-- Index pour trouver rapidement les colonnes avec default
CREATE INDEX IF NOT EXISTS idx_entity_fields_default
    ON entity_fields (entity_id)
    WHERE default_value IS NOT NULL;