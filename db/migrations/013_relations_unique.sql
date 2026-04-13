-- ============================================================
-- Migration 013 — Contrainte UNIQUE sur entity_relations
-- Fix accumulation de doublons entre syncs
-- OnePilot — 09/04/2026
-- ============================================================

-- Supprimer les doublons existants (garde le MIN(id))
DELETE FROM entity_relations
WHERE id NOT IN (
    SELECT MIN(id)
    FROM entity_relations
    GROUP BY source_id, source_entity, source_field,
             target_entity, COALESCE(target_field, '')
);

-- Créer l'index UNIQUE pour éviter les futurs doublons
CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_relations_unique
    ON entity_relations(
        source_id,
        source_entity,
        source_field,
        target_entity,
        COALESCE(target_field, '')
    );

-- ============================================================
-- Vérification
-- SELECT COUNT(*) FROM entity_relations; -- doit être < avant
-- ============================================================