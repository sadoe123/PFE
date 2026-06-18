"""
core/audit.py — Audit logs OnePilot
Écriture et lecture de audit_logs — isolé de la business logic
Phase 10 — Admin Console & Security
Correspond à CDC 2.6.1 : tous les accès aux données loggés (qui, quoi, quand)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Constantes actions ───────────────────────────────────────────────────────
class AuditAction:
    LOGIN          = "LOGIN"
    LOGOUT         = "LOGOUT"
    LOGIN_FAILED   = "LOGIN_FAILED"
    CREATE_USER    = "CREATE_USER"
    UPDATE_USER    = "UPDATE_USER"
    DELETE_USER    = "DELETE_USER"
    GRANT_PERM     = "GRANT_PERMISSION"
    REVOKE_PERM    = "REVOKE_PERMISSION"
    QUERY          = "QUERY"
    EXPORT         = "EXPORT"
    SYNC           = "SYNC"
    CONFIG_CHANGE  = "CONFIG_CHANGE"
    ACCESS_DENIED  = "ACCESS_DENIED"
    DASHBOARD_SAVE = "DASHBOARD_SAVE"
    VOICE_QUERY    = "VOICE_QUERY"


class AuditResult:
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR   = "error"


# ── Writer ───────────────────────────────────────────────────────────────────

async def log_event(
    action:     str,
    result:     str                  = AuditResult.SUCCESS,
    user_id:    Optional[int]        = None,
    user_email: Optional[str]        = None,
    resource:   Optional[str]        = None,
    details:    Optional[Dict]       = None,
    ip_address: Optional[str]        = None,
    user_agent: Optional[str]        = None,
) -> None:
    """
    Écrit un événement dans audit_logs.
    Non-bloquant : les erreurs sont loggées mais ne font pas planter la requête.
    """
    try:
        from api.database import get_pg_pool
        import json

        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO audit_logs
                    (user_id, user_email, action, resource, details,
                     ip_address, user_agent, result)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                user_id,
                user_email,
                action,
                resource,
                json.dumps(details) if details else None,
                ip_address,
                user_agent,
                result,
            )
    except Exception as e:
        logger.warning(f"[Audit] Échec écriture log ({action}): {e}")


# ── Reader ───────────────────────────────────────────────────────────────────

async def get_logs(
    limit:      int           = 50,
    offset:     int           = 0,
    user_email: Optional[str] = None,
    action:     Optional[str] = None,
    result:     Optional[str] = None,
    date_from:  Optional[str] = None,
    date_to:    Optional[str] = None,
) -> Dict[str, Any]:
    """
    Lit les audit logs avec filtres optionnels.
    Retourne {logs: [...], total: int}.
    """
    try:
        from api.database import get_pg_pool

        pool = await get_pg_pool()
        async with pool.acquire() as conn:

            # Construction dynamique du WHERE
            conditions = []
            params: List[Any] = []
            p = 1  # index paramètre PostgreSQL $1, $2...

            if user_email:
                conditions.append(f"user_email ILIKE ${p}")
                params.append(f"%{user_email}%")
                p += 1
            if action:
                conditions.append(f"action = ${p}")
                params.append(action.upper())
                p += 1
            if result:
                conditions.append(f"result = ${p}")
                params.append(result.lower())
                p += 1
            if date_from:
                conditions.append(f"created_at >= ${p}")
                params.append(date_from)
                p += 1
            if date_to:
                conditions.append(f"created_at <= ${p}")
                params.append(date_to)
                p += 1

            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

            # Total
            total = await conn.fetchval(
                f"SELECT COUNT(*) FROM audit_logs {where}", *params
            )

            # Logs paginés
            rows = await conn.fetch(
                f"""
                SELECT id, user_id, user_email, action, resource,
                       details, ip_address, user_agent, result,
                       created_at
                FROM audit_logs
                {where}
                ORDER BY created_at DESC
                LIMIT ${p} OFFSET ${p+1}
                """,
                *params, limit, offset
            )

            logs = [
                {
                    "id":         r["id"],
                    "user_id":    r["user_id"],
                    "user_email": r["user_email"],
                    "action":     r["action"],
                    "resource":   r["resource"],
                    "details":    r["details"],
                    "ip_address": r["ip_address"],
                    "user_agent": r["user_agent"],
                    "result":     r["result"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                }
                for r in rows
            ]

            return {"logs": logs, "total": total}

    except Exception as e:
        logger.error(f"[Audit] Erreur lecture logs: {e}")
        return {"logs": [], "total": 0}


async def get_stats() -> Dict[str, Any]:
    """Statistiques rapides pour le dashboard cockpit."""
    try:
        from api.database import get_pg_pool

        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            today = await conn.fetchval(
                "SELECT COUNT(*) FROM audit_logs WHERE created_at >= NOW() - INTERVAL '24 hours'"
            )
            failures = await conn.fetchval(
                "SELECT COUNT(*) FROM audit_logs WHERE result = 'failure' AND created_at >= NOW() - INTERVAL '24 hours'"
            )
            logins = await conn.fetchval(
                "SELECT COUNT(*) FROM audit_logs WHERE action = 'LOGIN' AND created_at >= NOW() - INTERVAL '24 hours'"
            )
            return {
                "events_today":   int(today   or 0),
                "failures_today": int(failures or 0),
                "logins_today":   int(logins   or 0),
            }
    except Exception as e:
        logger.error(f"[Audit] Erreur stats: {e}")
        return {"events_today": 0, "failures_today": 0, "logins_today": 0}
