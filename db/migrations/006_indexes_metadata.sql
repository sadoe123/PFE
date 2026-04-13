-- ============================================================
-- Migration 006 — Index DB + Objets DB avancés
-- OnePilot Layer 2 — Session 2 — 08/04/2026
-- ============================================================

-- 1. Colonne indexes sur source_entities
--    Stocke la liste des index d'une table/vue sous forme JSONB
--    ex: [{"name":"IX_foo","type":"NONCLUSTERED","unique":false,"columns":"col1, col2"}]
ALTER TABLE source_entities
    ADD COLUMN IF NOT EXISTS indexes  JSONB DEFAULT '[]';

-- 2. Colonne metadata sur source_entities
--    Stocke des métadonnées supplémentaires (definition_preview pour proc/func, etc.)
ALTER TABLE source_entities
    ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}';

-- 3. Colonne description sur entity_fields
--    Stocke les commentaires de colonnes (MS_Description / col_description PG)
ALTER TABLE entity_fields
    ADD COLUMN IF NOT EXISTS description TEXT;

-- 4. Index GIN pour requêtes JSONB rapides sur les indexes
CREATE INDEX IF NOT EXISTS idx_source_entities_indexes
    ON source_entities USING GIN (indexes)
    WHERE indexes IS NOT NULL AND indexes != '[]';

-- 5. Index sur entity_type pour filtrage rapide des objets DB avancés
CREATE INDEX IF NOT EXISTS idx_source_entities_type
    ON source_entities (source_id, entity_type);

-- ============================================================
-- Vérification
-- ============================================================
-- SELECT column_name, data_type
-- FROM information_schema.columns
-- WHERE table_name = 'source_entities'
--   AND column_name IN ('indexes', 'metadata');
--
-- SELECT column_name FROM information_schema.columns
-- WHERE table_name = 'entity_fields' AND column_name = 'description';