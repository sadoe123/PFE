-- ============================================================
-- Migration 010 — Synonymes métier spécifiques par source
-- OnePilot Layer 2 — §2.2.3.A — 09/04/2026
-- ============================================================

-- Table des synonymes personnalisés par source
CREATE TABLE IF NOT EXISTS source_synonyms (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id   UUID NOT NULL REFERENCES data_sources(id) ON DELETE CASCADE,
    term        VARCHAR(255) NOT NULL,       -- ex: "BALI", "SXORD"
    synonyms    TEXT[]       NOT NULL,       -- ex: ["balance", "solde", "account"]
    description TEXT,                        -- ex: "Balance comptable SXA"
    created_by  VARCHAR(255),
    created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE(source_id, term)
);

-- Index pour recherche rapide
CREATE INDEX IF NOT EXISTS idx_source_synonyms_source
    ON source_synonyms(source_id);

CREATE INDEX IF NOT EXISTS idx_source_synonyms_term
    ON source_synonyms(source_id, term);

-- ============================================================
-- Vérification
-- SELECT * FROM source_synonyms LIMIT 5;
-- ============================================================