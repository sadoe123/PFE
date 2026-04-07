-- ══════════════════════════════════════════════════════════════
-- MIGRATION 003: Table Conversations Chat IA
-- ══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES data_sources(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    messages JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_id UUID,  -- Pour multi-user future
    CONSTRAINT valid_messages CHECK (jsonb_typeof(messages) = 'array')
);

-- Index pour recherche rapide
CREATE INDEX IF NOT EXISTS idx_conversations_source_id ON conversations(source_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);

-- Trigger pour updated_at
CREATE OR REPLACE FUNCTION update_conversations_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS conversations_update_timestamp ON conversations;
CREATE TRIGGER conversations_update_timestamp
BEFORE UPDATE ON conversations
FOR EACH ROW
EXECUTE FUNCTION update_conversations_timestamp();
