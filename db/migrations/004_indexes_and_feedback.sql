-- ============================================================
-- OnePilot — Migration 004
-- Index confidence_score + full-text search + feedback loop
-- CORRIGÉ : relation_id BIGINT (compatible entity_relations.id bigint)
-- ============================================================

-- Extension trigram pour full-text search sur noms d'entités
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Index sur confidence (tri rapide, filtrage seuil)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entity_relations_confidence
    ON entity_relations(confidence DESC);

-- Index composite (source_entity, target_entity) pour path-finding
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entity_relations_path
    ON entity_relations(source_entity, target_entity);

-- Full-text search trigram sur les noms d'entités
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_source_entities_name_trgm
    ON source_entities USING gin(name gin_trgm_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_source_entities_display_trgm
    ON source_entities USING gin(COALESCE(display_name, name) gin_trgm_ops);

-- Full-text search trigram sur les champs
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entity_fields_name_trgm
    ON entity_fields USING gin(name gin_trgm_ops);

-- Index sur detection_method pour filtrer par algorithme
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entity_relations_method
    ON entity_relations(detection_method);

-- Index sur is_confirmed pour la validation humaine
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entity_relations_confirmed
    ON entity_relations(is_confirmed) WHERE is_confirmed IS NOT NULL;

-- ── Table feedback loop ML ────────────────────────────────────────────
-- CORRECTION : relation_id BIGINT pour correspondre à entity_relations.id (bigint)
-- Pas de FK car entity_relations.id peut être supprimé — on garde juste la référence
CREATE TABLE IF NOT EXISTS relation_feedback (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id       UUID    NOT NULL REFERENCES data_sources(id) ON DELETE CASCADE,
    relation_id     BIGINT,                    -- référence souple vers entity_relations.id (bigint)
    source_entity   VARCHAR(255) NOT NULL,
    source_field    VARCHAR(255) NOT NULL,
    target_entity   VARCHAR(255) NOT NULL,
    target_field    VARCHAR(255) NOT NULL,
    feedback        VARCHAR(20)  NOT NULL CHECK (feedback IN ('confirmed', 'rejected', 'alternative')),
    alternative_target_entity VARCHAR(255),
    alternative_target_field  VARCHAR(255),
    comment         TEXT,
    user_id         VARCHAR(255),
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_relation_feedback_source
    ON relation_feedback(source_id);

CREATE INDEX IF NOT EXISTS idx_relation_feedback_feedback
    ON relation_feedback(feedback);

CREATE INDEX IF NOT EXISTS idx_relation_feedback_relation
    ON relation_feedback(relation_id) WHERE relation_id IS NOT NULL;

-- ── Table graph cache metadata ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS relation_graph_meta (
    source_id   UUID PRIMARY KEY REFERENCES data_sources(id) ON DELETE CASCADE,
    node_count  INTEGER NOT NULL DEFAULT 0,
    edge_count  INTEGER NOT NULL DEFAULT 0,
    cycle_count INTEGER NOT NULL DEFAULT 0,
    cycles      JSONB   NOT NULL DEFAULT '[]',
    computed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- ── Colonnes additionnelles sur entity_relations (si absentes) ────────
ALTER TABLE entity_relations
    ADD COLUMN IF NOT EXISTS validated_by  VARCHAR(255),
    ADD COLUMN IF NOT EXISTS validated_at  TIMESTAMP WITH TIME ZONE,
    ADD COLUMN IF NOT EXISTS reject_reason TEXT,
    ADD COLUMN IF NOT EXISTS value_overlap FLOAT,
    ADD COLUMN IF NOT EXISTS source_entity VARCHAR(255),
    ADD COLUMN IF NOT EXISTS target_entity VARCHAR(255);

-- ── Index sur data_sources pour la recherche cross-sources ───────────
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_data_sources_name_trgm
    ON data_sources USING gin(name gin_trgm_ops);