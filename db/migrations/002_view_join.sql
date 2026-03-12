-- ============================================================
-- Migration 002 : Import Vues SQL — relations view_join
-- Ajoute les colonnes manquantes à entity_relations
-- ============================================================

-- Colonne detection_method (explicit_fk, view_join, name_pascal, fuzzy, etc.)
ALTER TABLE entity_relations
    ADD COLUMN IF NOT EXISTS detection_method VARCHAR(50) DEFAULT 'explicit_fk';

-- Colonne view_name : nom de la vue source pour les relations view_join
ALTER TABLE entity_relations
    ADD COLUMN IF NOT EXISTS view_name VARCHAR(500);

-- Colonne source_id : lien direct vers data_sources (plus facile à requêter)
-- (certaines versions n'ont que source_entity_id via source_entities)
ALTER TABLE entity_relations
    ADD COLUMN IF NOT EXISTS source_id UUID REFERENCES data_sources(id) ON DELETE CASCADE;

-- Colonnes source_entity / target_entity (noms directs, sans passer par les UUIDs)
ALTER TABLE entity_relations
    ADD COLUMN IF NOT EXISTS source_entity VARCHAR(500);
ALTER TABLE entity_relations
    ADD COLUMN IF NOT EXISTS target_entity VARCHAR(500);

-- Colonne reject_reason pour la validation humaine
ALTER TABLE entity_relations
    ADD COLUMN IF NOT EXISTS reject_reason TEXT;

-- Colonne validated_by
ALTER TABLE entity_relations
    ADD COLUMN IF NOT EXISTS validated_by VARCHAR(100);

-- Index pour les requêtes fréquentes
CREATE INDEX IF NOT EXISTS idx_entity_relations_source_id
    ON entity_relations(source_id);

CREATE INDEX IF NOT EXISTS idx_entity_relations_method
    ON entity_relations(source_id, detection_method);

CREATE INDEX IF NOT EXISTS idx_entity_relations_view_join
    ON entity_relations(source_id, view_name)
    WHERE detection_method = 'view_join';

CREATE INDEX IF NOT EXISTS idx_entity_relations_entities
    ON entity_relations(source_id, source_entity, target_entity);

-- Mettre à jour les relations existantes (explicit_fk déjà présentes)
UPDATE entity_relations
SET detection_method = 'explicit_fk'
WHERE detection_method IS NULL;