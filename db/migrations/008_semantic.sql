-- ============================================================
-- Migration 008 — Indexation sémantique
-- OnePilot Layer 2 — §2.2.3 — 08/04/2026
-- ============================================================

-- 1. Extension pgvector pour les embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Colonnes sémantiques sur source_entities
ALTER TABLE source_entities
    ADD COLUMN IF NOT EXISTS business_domain    VARCHAR(50),   -- Finance, RH, Ventes, Logistique...
    ADD COLUMN IF NOT EXISTS business_concept   VARCHAR(50),   -- Customer, Order, Invoice, Product...
    ADD COLUMN IF NOT EXISTS entity_class       VARCHAR(20),   -- transactional, reference, log, config
    ADD COLUMN IF NOT EXISTS semantic_tags      JSONB DEFAULT '[]',  -- tags métier
    ADD COLUMN IF NOT EXISTS dimensions         JSONB DEFAULT '{}',  -- {time: [...], geo: [...], product: [...]}
    ADD COLUMN IF NOT EXISTS embedding          vector(384);   -- vecteur TF-IDF 384 dims

-- 3. Colonnes sémantiques sur entity_fields  
ALTER TABLE entity_fields
    ADD COLUMN IF NOT EXISTS dimension_type     VARCHAR(20),   -- time, geo, product, amount, id, status
    ADD COLUMN IF NOT EXISTS semantic_concept   VARCHAR(50);   -- date_creation, montant_ttc, code_pays...

-- 4. Index vectoriel pour recherche sémantique (IVFFlat)
CREATE INDEX IF NOT EXISTS idx_source_entities_embedding
    ON source_entities USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10)
    WHERE embedding IS NOT NULL;

-- 5. Index GIN sur semantic_tags
CREATE INDEX IF NOT EXISTS idx_source_entities_tags
    ON source_entities USING GIN (semantic_tags);

-- 6. Index sur business_domain pour filtrage
CREATE INDEX IF NOT EXISTS idx_source_entities_domain
    ON source_entities (source_id, business_domain)
    WHERE business_domain IS NOT NULL;

-- ============================================================
-- Vérification
-- SELECT column_name FROM information_schema.columns
-- WHERE table_name = 'source_entities'
--   AND column_name IN ('business_domain','business_concept','embedding');
-- ============================================================