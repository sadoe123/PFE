-- Migration 022 : table password_reset_tokens pour reset lien admin
-- Sprint S2 — Users RBAC complet

CREATE TABLE IF NOT EXISTS password_reset_tokens (
    user_id    INT PRIMARY KEY REFERENCES op_users(id) ON DELETE CASCADE,
    token      VARCHAR(64) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prt_token ON password_reset_tokens(token);
