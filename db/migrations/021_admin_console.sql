-- ============================================================
-- Migration 012 — Admin Console & Security (Phase 10)
-- OnePilot v0 — 2026-06-15
-- Tables : op_users, user_source_permissions, audit_logs, admin_config
-- ============================================================

-- ── 1. Utilisateurs OnePilot ────────────────────────────────
CREATE TABLE IF NOT EXISTS op_users (
    id            SERIAL PRIMARY KEY,
    email         TEXT        NOT NULL UNIQUE,
    username      TEXT        NOT NULL,
    password_hash TEXT        NOT NULL,
    role          TEXT        NOT NULL DEFAULT 'user',
    -- Rôles valides : 'admin' | 'power_user' | 'user'
    is_active     BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_op_users_email
    ON op_users(email);

CREATE INDEX IF NOT EXISTS idx_op_users_role
    ON op_users(role) WHERE is_active = TRUE;

-- Trigger : met à jour updated_at automatiquement
CREATE OR REPLACE FUNCTION update_op_users_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_op_users_updated_at ON op_users;
CREATE TRIGGER trg_op_users_updated_at
    BEFORE UPDATE ON op_users
    FOR EACH ROW EXECUTE FUNCTION update_op_users_updated_at();


-- ── 2. Permissions par source de données ────────────────────
-- Correspond à CDC 2.6.1.B — Row-level security par source
CREATE TABLE IF NOT EXISTS user_source_permissions (
    id         SERIAL      PRIMARY KEY,
    user_id    INTEGER     NOT NULL REFERENCES op_users(id) ON DELETE CASCADE,
    source_id  TEXT        NOT NULL,  -- UUID source (TEXT pour compatibilité)
    can_read   BOOLEAN     NOT NULL DEFAULT TRUE,
    can_export BOOLEAN     NOT NULL DEFAULT FALSE,
    can_query  BOOLEAN     NOT NULL DEFAULT TRUE,
    granted_by INTEGER     REFERENCES op_users(id),
    granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, source_id)
);

CREATE INDEX IF NOT EXISTS idx_usp_user_id
    ON user_source_permissions(user_id);

CREATE INDEX IF NOT EXISTS idx_usp_source_id
    ON user_source_permissions(source_id);


-- ── 3. Audit Logs ───────────────────────────────────────────
-- Correspond à CDC 2.6.1 — tous les accès loggés
CREATE TABLE IF NOT EXISTS audit_logs (
    id         SERIAL      PRIMARY KEY,
    user_id    INTEGER     REFERENCES op_users(id) ON DELETE SET NULL,
    user_email TEXT,
    action     TEXT        NOT NULL,
    -- Actions : LOGIN | LOGOUT | CREATE_USER | DELETE_USER |
    --           QUERY | EXPORT | SYNC | CONFIG_CHANGE | ACCESS_DENIED
    resource   TEXT,
    -- Ex: 'source:uuid', 'user:42', 'dashboard:xxx'
    details    JSONB,
    ip_address TEXT,
    user_agent TEXT,
    result     TEXT        NOT NULL DEFAULT 'success',
    -- Résultats : 'success' | 'failure' | 'error'
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_created
    ON audit_logs(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_user_email
    ON audit_logs(user_email, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_action
    ON audit_logs(action, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_result
    ON audit_logs(result) WHERE result != 'success';


-- ── 4. Configuration Admin ──────────────────────────────────
-- Correspond à CDC 2.6.1.C (Voice), 2.6.1.D (Dashboards)
-- Stockage clé/valeur JSON flexible
CREATE TABLE IF NOT EXISTS admin_config (
    key        TEXT        PRIMARY KEY,
    value      JSONB       NOT NULL,
    updated_by INTEGER     REFERENCES op_users(id) ON DELETE SET NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Valeurs par défaut de configuration
INSERT INTO admin_config (key, value) VALUES
    ('voice_stt_engine',       '"whisper"'),
    ('voice_tts_engine',       '"piper"'),
    ('voice_default_lang',     '"fr"'),
    ('voice_retain_audio',     'false'),
    ('voice_retention_days',   '7'),
    ('voice_anonymize',        'false'),
    ('dashboard_max_widgets',  '20'),
    ('dashboard_max_export',   '10000'),
    ('dashboard_refresh_sec',  '300'),
    ('session_expire_hours',   '24')
ON CONFLICT (key) DO NOTHING;


-- ── 5. Vérification de cohérence ────────────────────────────
-- S'assure que les colonnes existent (idempotent)
ALTER TABLE op_users
    ADD COLUMN IF NOT EXISTS last_login TIMESTAMPTZ;

ALTER TABLE user_source_permissions
    ADD COLUMN IF NOT EXISTS can_query BOOLEAN NOT NULL DEFAULT TRUE;

ALTER TABLE audit_logs
    ADD COLUMN IF NOT EXISTS user_agent TEXT;
