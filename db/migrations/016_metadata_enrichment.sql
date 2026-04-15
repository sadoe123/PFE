-- ══════════════════════════════════════════════════════════════
-- OnePilot §2.2.1/2.2.2/2.2.5 completion — Migration 016
-- Métadonnées enrichies : descriptions, dépendances, cardinalité
-- ══════════════════════════════════════════════════════════════

-- ── Colonnes enrichissement sur source_entities ───────────────
ALTER TABLE source_entities
    ADD COLUMN IF NOT EXISTS metadata    JSONB  DEFAULT NULL;

-- ── Table log WAL CDC ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cdc_wal_log (
    id          UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id   UUID         NOT NULL REFERENCES data_sources(id) ON DELETE CASCADE,
    event_type  VARCHAR(50)  NOT NULL,
    object_name VARCHAR(255),
    lsn         VARCHAR(64),
    payload     JSONB        NOT NULL DEFAULT '{}',
    detected_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_wal_source ON cdc_wal_log (source_id, detected_at DESC);

-- ── Table log HATEOAS relations ───────────────────────────────
-- Les relations HATEOAS vont dans entity_relations (déjà existant)
-- On ajoute juste un index sur detection_method pour filtrer
CREATE INDEX IF NOT EXISTS idx_er_detection_method
    ON entity_relations (source_id, detection_method)
    WHERE detection_method IN ('graphql_nested_type','hateoas_ref','hateoas_hal','hateoas_uri_field');

-- ── Vue : résumé enrichissement par source ────────────────────
CREATE OR REPLACE VIEW source_enrichment_summary AS
SELECT
    ds.id                                      AS source_id,
    ds.name                                    AS source_name,
    ds.connector_type,
    COUNT(DISTINCT se.id)                      AS total_entities,
    COUNT(DISTINCT se.id) FILTER (
        WHERE se.description IS NOT NULL
        AND   se.description != ''
    )                                          AS entities_with_description,
    COUNT(DISTINCT se.id) FILTER (
        WHERE se.metadata->>'dependencies' IS NOT NULL
    )                                          AS entities_with_dependencies,
    COUNT(DISTINCT se.id) FILTER (
        WHERE se.row_count IS NOT NULL
    )                                          AS entities_with_rowcount,
    COUNT(DISTINCT er.id) FILTER (
        WHERE er.detection_method IN ('graphql_nested_type','hateoas_ref','hateoas_hal')
    )                                          AS hateoas_graphql_relations
FROM   data_sources ds
LEFT JOIN source_entities  se ON se.source_id = ds.id
LEFT JOIN entity_relations er ON er.source_id = ds.id
GROUP BY ds.id, ds.name, ds.connector_type;
