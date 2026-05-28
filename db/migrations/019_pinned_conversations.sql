-- Migration 019 : Conversations épinglées
-- Permet d'épingler directement une conversation (Option A)
-- indépendamment des dashboards favoris

CREATE TABLE IF NOT EXISTS pinned_conversations (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conv_id     UUID NOT NULL,
    user_id     TEXT NOT NULL DEFAULT 'admin',
    pinned_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(conv_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_pinned_conv_user
    ON pinned_conversations(user_id, conv_id);

CREATE INDEX IF NOT EXISTS idx_pinned_conv_id
    ON pinned_conversations(conv_id);
