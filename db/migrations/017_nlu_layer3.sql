-- ══════════════════════════════════════════════════════════════
-- OnePilot Layer 3 §2.3 — Migration 017
-- NLU Engine : sessions, query plans, clarifications
-- ══════════════════════════════════════════════════════════════

-- ── Sessions NLU (contexte multi-tour par conversation) ──────
CREATE TABLE IF NOT EXISTS nlu_sessions (
    id              UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id VARCHAR(255) NOT NULL,
    source_id       UUID    REFERENCES data_sources(id) ON DELETE SET NULL,
    last_intent     VARCHAR(100),
    last_table      VARCHAR(255),
    last_field      VARCHAR(255),
    turn_count      INTEGER DEFAULT 0,
    context_json    JSONB   DEFAULT '{}',
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_nlu_session_conv ON nlu_sessions (conversation_id);

-- ── Log des requêtes NLU ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS nlu_query_log (
    id              UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id VARCHAR(255),
    source_id       UUID    REFERENCES data_sources(id) ON DELETE SET NULL,
    question        TEXT    NOT NULL,
    intent          VARCHAR(100),
    confidence      FLOAT,
    tables_detected TEXT[],
    slots_json      JSONB   DEFAULT '{}',
    sql_generated   TEXT,
    needs_clarification BOOLEAN DEFAULT FALSE,
    clarification_json  JSONB,
    response_ms     INTEGER,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_nlu_log_source ON nlu_query_log (source_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_nlu_log_intent ON nlu_query_log (intent);

-- ── Plans de requête cross-source ────────────────────────────
CREATE TABLE IF NOT EXISTS query_plans (
    id              UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    question        TEXT    NOT NULL,
    source_ids      UUID[]  NOT NULL,
    plan_json       JSONB   NOT NULL,
    result_json     JSONB,
    status          VARCHAR(50) DEFAULT 'pending',
    execution_ms    INTEGER,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ── Vue : stats NLU par intent ────────────────────────────────
CREATE OR REPLACE VIEW nlu_intent_stats AS
SELECT
    intent,
    COUNT(*)                            AS total_queries,
    AVG(confidence)                     AS avg_confidence,
    AVG(response_ms)                    AS avg_response_ms,
    SUM(CASE WHEN needs_clarification THEN 1 ELSE 0 END) AS clarifications,
    MAX(created_at)                     AS last_used
FROM nlu_query_log
WHERE intent IS NOT NULL
GROUP BY intent
ORDER BY total_queries DESC;