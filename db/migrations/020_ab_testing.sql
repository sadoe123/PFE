-- ============================================================
-- Migration 020 : A/B Testing de prompts
-- OnePilot — Sprint 13
-- Date : 2026-05-27
-- Description :
--   Système d'expérimentation pour comparer deux variantes
--   de prompts LLM (Prompt A baseline vs Prompt B few-shot).
--   Chaque appel LLM enregistre le variant utilisé, le SQL
--   généré, sa validité, et un score de qualité.
-- ============================================================

-- ── Table principale des tests A/B ───────────────────────────
CREATE TABLE IF NOT EXISTS ab_test_results (
    id              SERIAL          PRIMARY KEY,
    question        TEXT            NOT NULL,
    source_id       UUID,
    prompt_variant  CHAR(1)         NOT NULL CHECK (prompt_variant IN ('A', 'B')),
    sql_generated   TEXT,
    sql_valid       BOOLEAN         NOT NULL DEFAULT FALSE,
    has_results     BOOLEAN         NOT NULL DEFAULT FALSE,
    row_count       INTEGER         NOT NULL DEFAULT 0,
    error_message   TEXT,
    score           FLOAT           NOT NULL DEFAULT 0,
    winner          BOOLEAN         NOT NULL DEFAULT FALSE,
    duration_ms     INTEGER         NOT NULL DEFAULT 0,
    few_shot_count  INTEGER         NOT NULL DEFAULT 0,   -- nb exemples injectés (Prompt B)
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- ── Index sur source + date (stats par source) ───────────────
CREATE INDEX IF NOT EXISTS idx_ab_test_source_date
    ON ab_test_results(source_id, created_at DESC);

-- ── Index sur variant + validité (comparaison A vs B) ────────
CREATE INDEX IF NOT EXISTS idx_ab_test_variant_valid
    ON ab_test_results(prompt_variant, sql_valid);

-- ── Index sur winner (requêtes rapides pour dashboard) ───────
CREATE INDEX IF NOT EXISTS idx_ab_test_winner
    ON ab_test_results(winner, created_at DESC);

-- ── Vue agrégée pour le dashboard stats ──────────────────────
CREATE OR REPLACE VIEW ab_test_stats AS
SELECT
    prompt_variant,
    COUNT(*)                                        AS total_tests,
    COUNT(*) FILTER (WHERE sql_valid = TRUE)        AS valid_sql_count,
    COUNT(*) FILTER (WHERE has_results = TRUE)      AS has_results_count,
    COUNT(*) FILTER (WHERE winner = TRUE)           AS wins,
    ROUND(AVG(score)::NUMERIC, 3)                   AS avg_score,
    ROUND(AVG(duration_ms)::NUMERIC, 0)             AS avg_duration_ms,
    ROUND(
        (COUNT(*) FILTER (WHERE sql_valid = TRUE)::FLOAT
        / NULLIF(COUNT(*), 0) * 100)::NUMERIC, 1
    )                                               AS valid_pct,
    ROUND(
        (COUNT(*) FILTER (WHERE has_results = TRUE)::FLOAT
        / NULLIF(COUNT(*), 0) * 100)::NUMERIC, 1
    )                                               AS results_pct,
    MAX(created_at)                                 AS last_test_at
FROM ab_test_results
GROUP BY prompt_variant
ORDER BY prompt_variant;

-- ── Commentaires documentation ────────────────────────────────
COMMENT ON TABLE ab_test_results IS
    'Résultats des tests A/B de prompts LLM — Sprint 13 OnePilot';
COMMENT ON COLUMN ab_test_results.prompt_variant IS
    'A = Prompt baseline | B = Prompt enrichi few-shot';
COMMENT ON COLUMN ab_test_results.score IS
    'Score qualité : sql_valid(1pt) + has_results(1pt) + row_count>0(1pt) = max 3';
COMMENT ON COLUMN ab_test_results.winner IS
    'TRUE si ce variant a été retourné à l utilisateur lors d un test simultané';
COMMENT ON COLUMN ab_test_results.few_shot_count IS
    'Nombre d exemples few-shot injectés dans le Prompt B (0 pour Prompt A)';
