-- OnePilot – Migration 018 §2.3.4 — user_preferences
-- Apprentissage des préférences utilisateur

-- Table principale des préférences
CREATE TABLE IF NOT EXISTS user_preferences (
    id              BIGSERIAL PRIMARY KEY,
    user_id         VARCHAR(255)  NOT NULL DEFAULT 'default',
    source_id       VARCHAR(255)  NOT NULL DEFAULT '',
    slot_key        VARCHAR(100)  NOT NULL,
    value           VARCHAR(500)  NOT NULL,
    usage_count     INTEGER       NOT NULL DEFAULT 1,
    last_used       TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    question_sample TEXT,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_user_pref UNIQUE (user_id, source_id, slot_key, value)
);

-- Index pour les lookups fréquents
CREATE INDEX IF NOT EXISTS idx_user_pref_lookup
    ON user_preferences (user_id, slot_key, value);

CREATE INDEX IF NOT EXISTS idx_user_pref_source
    ON user_preferences (user_id, source_id);

CREATE INDEX IF NOT EXISTS idx_user_pref_usage
    ON user_preferences (usage_count DESC);

-- Vue agrégée des préférences les plus utilisées
CREATE OR REPLACE VIEW top_user_preferences AS
SELECT
    user_id,
    source_id,
    slot_key,
    value,
    usage_count,
    last_used,
    ROUND(usage_count::numeric /
        NULLIF(SUM(usage_count) OVER (PARTITION BY user_id, slot_key), 0), 3
    ) AS preference_score
FROM user_preferences
ORDER BY user_id, slot_key, usage_count DESC;