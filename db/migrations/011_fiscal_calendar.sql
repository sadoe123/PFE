-- ============================================================
-- Migration 011 — Calendriers fiscaux par source
-- OnePilot Layer 2 — §2.2.3.B — 09/04/2026
-- ============================================================

CREATE TABLE IF NOT EXISTS source_fiscal_calendars (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id           UUID NOT NULL REFERENCES data_sources(id) ON DELETE CASCADE,
    fiscal_year_start   INTEGER NOT NULL DEFAULT 1  -- mois de début (1=Jan, 7=Juil, 10=Oct)
        CHECK (fiscal_year_start BETWEEN 1 AND 12),
    fiscal_year_label   VARCHAR(50) DEFAULT 'FY',   -- ex: "FY", "EX", "Exercice"
    description         TEXT,
    created_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE(source_id)
);

CREATE INDEX IF NOT EXISTS idx_fiscal_calendars_source
    ON source_fiscal_calendars(source_id);

-- ============================================================
-- Exemples :
-- fiscal_year_start=1  → Janvier  (standard)
-- fiscal_year_start=4  → Avril    (UK fiscal year)
-- fiscal_year_start=7  → Juillet  (courant en Afrique du Nord)
-- fiscal_year_start=10 → Octobre  (US federal)
-- ============================================================