-- ============================================================
-- Migration 012 — Taxonomie personnalisable + Historique recherches
-- OnePilot Layer 2 — §2.2.3 — 09/04/2026
-- ============================================================

-- ── TABLE 1 : Domaines personnalisés par source ───────────────
-- Permet au client d'ajouter ses propres domaines métier
-- ex: "Trésorerie", "Qualité", "Maintenance"
CREATE TABLE IF NOT EXISTS source_domains (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id   UUID NOT NULL REFERENCES data_sources(id) ON DELETE CASCADE,
    domain_name VARCHAR(100) NOT NULL,       -- ex: "Trésorerie"
    patterns    TEXT[] NOT NULL DEFAULT '{}', -- ex: ["tresor","cash","bnk","bank"]
    color       VARCHAR(20) DEFAULT '#6366f1', -- couleur UI
    icon        VARCHAR(50) DEFAULT 'database', -- icône UI
    description TEXT,
    priority    INTEGER DEFAULT 0,           -- priorité vs domaines par défaut
    created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE(source_id, domain_name)
);

CREATE INDEX IF NOT EXISTS idx_source_domains_source
    ON source_domains(source_id);

-- ── TABLE 2 : Historique des recherches sémantiques ──────────
-- Enregistre chaque recherche + résultats pour boosting futur
CREATE TABLE IF NOT EXISTS search_history (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id    UUID REFERENCES data_sources(id) ON DELETE CASCADE,
    query        TEXT NOT NULL,              -- ex: "facture client"
    query_norm   TEXT NOT NULL,              -- query normalisée (lowercase, trim)
    results      JSONB NOT NULL DEFAULT '[]', -- top 5 résultats retournés
    result_count INTEGER DEFAULT 0,
    clicked_id   UUID,                       -- entité cliquée par l'utilisateur
    clicked_name VARCHAR(255),
    user_id      VARCHAR(255),
    search_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_search_history_source
    ON search_history(source_id);

CREATE INDEX IF NOT EXISTS idx_search_history_query
    ON search_history(query_norm);

CREATE INDEX IF NOT EXISTS idx_search_history_clicked
    ON search_history(clicked_id) WHERE clicked_id IS NOT NULL;

-- Vue agrégée : top requêtes par source
CREATE OR REPLACE VIEW search_history_stats AS
SELECT
    source_id,
    query_norm,
    COUNT(*)                                    AS search_count,
    COUNT(clicked_id)                           AS click_count,
    ROUND(COUNT(clicked_id)::numeric /
          NULLIF(COUNT(*), 0) * 100, 1)         AS click_rate_pct,
    MAX(search_at)                              AS last_searched_at,
    MODE() WITHIN GROUP (ORDER BY clicked_name) AS most_clicked
FROM search_history
GROUP BY source_id, query_norm
ORDER BY search_count DESC;

-- ============================================================
-- Vérification
-- SELECT * FROM source_domains LIMIT 5;
-- SELECT * FROM search_history_stats LIMIT 5;
-- ============================================================