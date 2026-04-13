-- ══════════════════════════════════════════════════════════════
-- OnePilot §2.2.5 — CDC / Versioning
-- Migration à appliquer après 001_init.sql
-- ══════════════════════════════════════════════════════════════

-- ── Table principale : historique des versions de schéma ─────

CREATE TABLE IF NOT EXISTS schema_versions (
    id                    UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id             UUID         NOT NULL REFERENCES data_sources(id) ON DELETE CASCADE,
    version_number        INTEGER      NOT NULL,
    fingerprint           VARCHAR(64)  NOT NULL,

    -- Snapshot complet du schéma au moment de la version
    schema_snapshot       JSONB        NOT NULL,

    -- Delta par rapport à la version précédente (liste de changements)
    changes_delta         JSONB        NOT NULL DEFAULT '[]',

    -- Résumé lisible (counts par type, tables affectées)
    change_summary        JSONB        NOT NULL DEFAULT '{}',

    -- Flags
    has_breaking_changes  BOOLEAN      NOT NULL DEFAULT FALSE,
    is_rollback           BOOLEAN      NOT NULL DEFAULT FALSE,
    rollback_from_version INTEGER,

    created_at            TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    UNIQUE (source_id, version_number)
);

-- ── Table tags Git-like ───────────────────────────────────────

CREATE TABLE IF NOT EXISTS schema_version_tags (
    id             UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id      UUID         NOT NULL REFERENCES data_sources(id) ON DELETE CASCADE,
    version_number INTEGER      NOT NULL,
    tag            VARCHAR(128) NOT NULL,
    note           TEXT         DEFAULT '',
    created_at     TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    UNIQUE (source_id, tag)
);

-- ── Index de performance ──────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_sv_source_version
    ON schema_versions (source_id, version_number DESC);

CREATE INDEX IF NOT EXISTS idx_sv_breaking
    ON schema_versions (source_id, has_breaking_changes)
    WHERE has_breaking_changes = TRUE;

CREATE INDEX IF NOT EXISTS idx_sv_created
    ON schema_versions (source_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_svt_source
    ON schema_version_tags (source_id, version_number);

-- ── Vue utilitaire : dernière version par source ──────────────

CREATE OR REPLACE VIEW schema_versions_latest AS
SELECT DISTINCT ON (source_id)
    source_id,
    version_number,
    fingerprint,
    has_breaking_changes,
    is_rollback,
    change_summary,
    created_at
FROM   schema_versions
ORDER  BY source_id, version_number DESC;

-- ── Colonne indexes sur source_entities (si absente) ─────────
-- Utilisée par le snapshot CDC pour récupérer les index existants

ALTER TABLE source_entities
    ADD COLUMN IF NOT EXISTS indexes  JSONB  DEFAULT NULL;

ALTER TABLE source_entities
    ADD COLUMN IF NOT EXISTS metadata JSONB  DEFAULT NULL;
    