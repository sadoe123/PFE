-- ══════════════════════════════════════════════════════════════
-- OnePilot §2.2.5 — CDC File Watcher + Auto-Reindexer
-- Migration 015
-- ══════════════════════════════════════════════════════════════

-- ── Historique CDC fichiers ───────────────────────────────────
CREATE TABLE IF NOT EXISTS file_cdc_history (
    id               UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id        UUID         NOT NULL REFERENCES data_sources(id) ON DELETE CASCADE,
    filepath         TEXT         NOT NULL,
    checksum         VARCHAR(64)  NOT NULL,
    size_bytes       BIGINT       NOT NULL,
    file_modified_at TEXT,
    change_type      VARCHAR(20)  NOT NULL DEFAULT 'MODIFIED',
    size_delta       BIGINT,
    detected_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fch_source ON file_cdc_history (source_id, detected_at DESC);

-- ── Log réindexation CDC ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS cdc_reindex_log (
    id                   UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id            UUID         NOT NULL REFERENCES data_sources(id) ON DELETE CASCADE,
    schema_version       INTEGER,
    deleted_count        INTEGER      NOT NULL DEFAULT 0,
    reindexed_count      INTEGER      NOT NULL DEFAULT 0,
    relations_preserved  INTEGER      NOT NULL DEFAULT 0,
    errors               JSONB        NOT NULL DEFAULT '[]',
    created_at           TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_crl_source ON cdc_reindex_log (source_id, created_at DESC);

-- ── Colonnes needs_review sur entity_relations ────────────────
ALTER TABLE entity_relations
    ADD COLUMN IF NOT EXISTS needs_review  BOOLEAN   NOT NULL DEFAULT FALSE;

ALTER TABLE entity_relations
    ADD COLUMN IF NOT EXISTS review_reason TEXT      DEFAULT NULL;

ALTER TABLE entity_relations
    ADD COLUMN IF NOT EXISTS reviewed_at   TIMESTAMP WITH TIME ZONE DEFAULT NULL;

CREATE INDEX IF NOT EXISTS idx_er_needs_review
    ON entity_relations (source_id, needs_review)
    WHERE needs_review = TRUE;

-- ── Vue : relations nécessitant une révision ──────────────────
CREATE OR REPLACE VIEW entity_relations_to_review AS
SELECT
    er.id,
    er.source_id,
    er.source_entity,
    er.source_field,
    er.target_entity,
    er.target_field,
    er.relation_type,
    er.confidence,
    er.review_reason,
    er.created_at
FROM   entity_relations er
WHERE  er.needs_review = TRUE
  AND  er.reviewed_at  IS NULL
ORDER  BY er.source_id, er.confidence DESC;