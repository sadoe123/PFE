"""
routes/admin.py — Console Admin OnePilot
Endpoints : users CRUD, permissions sources, audit logs, config
Phase 10 — Admin Console & Security
Correspond à CDC 2.6.1.B (Users/RBAC), 2.6.1.C (Voice), 2.6.1.D (Dashboards)
Tous les endpoints nécessitent role=admin
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin Console"])


# ── Schémas Pydantic ─────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    email:    str
    username: str
    password: str
    role:     str = "user"   # admin | power_user | user
    is_active: bool = True


class UserUpdate(BaseModel):
    username:  Optional[str]  = None
    role:      Optional[str]  = None
    is_active: Optional[bool] = None
    password:  Optional[str]  = None   # None = ne pas changer


class SourcePermissionSet(BaseModel):
    source_id:  str
    can_read:   bool = True
    can_export: bool = False
    can_query:  bool = True


class ConfigUpdate(BaseModel):
    key:   str
    value: Any


# ── Helper : admin requis ─────────────────────────────────────────────────────

async def _admin(request: Request) -> Dict[str, Any]:
    """Vérifie JWT + role admin sur chaque requête admin."""
    from core.auth import extract_token_from_header, decode_token
    token = extract_token_from_header(request.headers.get("authorization"))
    if not token:
        raise HTTPException(401, "Token manquant")
    payload = decode_token(token)
    if not payload:
        raise HTTPException(401, "Token invalide ou expiré")
    if payload.get("role") != "admin":
        raise HTTPException(403, "Accès réservé aux administrateurs")
    return payload


# ════════════════════════════════════════════════════════════
# USERS CRUD — CDC 2.6.1.B
# ════════════════════════════════════════════════════════════

@router.get("/users")
async def list_users(
    request: Request,
    page:     int = Query(1,  ge=1),
    per_page: int = Query(20, ge=1, le=100),
    _admin = Depends(_admin),
):
    """Liste tous les utilisateurs avec pagination."""
    from api.database import get_pg_pool

    offset = (page - 1) * per_page
    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM op_users")
            rows  = await conn.fetch(
                """
                SELECT id, email, username, role, is_active,
                       created_at, updated_at, last_login
                FROM op_users
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
                """,
                per_page, offset
            )
        users = [
            {
                "id":         r["id"],
                "email":      r["email"],
                "username":   r["username"],
                "role":       r["role"],
                "is_active":  r["is_active"],
                "created_at": r["created_at"].isoformat()  if r["created_at"]  else None,
                "updated_at": r["updated_at"].isoformat()  if r["updated_at"]  else None,
                "last_login": r["last_login"].isoformat()  if r["last_login"]  else None,
            }
            for r in rows
        ]
        return {"users": users, "total": total, "page": page, "per_page": per_page}
    except Exception as e:
        logger.error(f"[Admin/users] list: {e}")
        raise HTTPException(500, "Erreur base de données")


@router.post("/users", status_code=201)
async def create_user(body: UserCreate, request: Request, caller=Depends(_admin)):
    """Crée un nouvel utilisateur. Vérifie unicité email."""
    from api.database import get_pg_pool
    from core.auth   import hash_password
    from core.audit  import log_event, AuditAction

    VALID_ROLES = {"admin", "power_user", "user"}
    if body.role not in VALID_ROLES:
        raise HTTPException(400, f"Rôle invalide. Valides: {VALID_ROLES}")

    try:
        pwd_hash = hash_password(body.password)
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            # Vérifier unicité
            existing = await conn.fetchval(
                "SELECT id FROM op_users WHERE email = $1",
                body.email.lower().strip()
            )
            if existing:
                raise HTTPException(409, "Email déjà utilisé")

            user_id = await conn.fetchval(
                """
                INSERT INTO op_users (email, username, password_hash, role, is_active)
                VALUES ($1, $2, $3, $4, $5) RETURNING id
                """,
                body.email.lower().strip(),
                body.username,
                pwd_hash,
                body.role,
                body.is_active,
            )

        await log_event(
            action=AuditAction.CREATE_USER,
            user_id=int(caller["sub"]),
            user_email=caller["email"],
            resource=f"user:{user_id}",
            details={"email": body.email, "role": body.role},
        )
        logger.info(f"[Admin] Utilisateur créé: {body.email} ({body.role})")
        return {"id": user_id, "email": body.email, "role": body.role}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Admin/users] create: {e}")
        raise HTTPException(500, "Erreur création utilisateur")


@router.put("/users/{user_id}")
async def update_user(
    user_id: int, body: UserUpdate, request: Request, caller=Depends(_admin)
):
    """Modifie un utilisateur (username, role, is_active, password)."""
    from api.database import get_pg_pool
    from core.auth   import hash_password
    from core.audit  import log_event, AuditAction

    updates   = []
    params: List[Any] = []
    p = 1

    if body.username is not None:
        updates.append(f"username = ${p}"); params.append(body.username); p += 1
    if body.role is not None:
        VALID_ROLES = {"admin", "power_user", "user"}
        if body.role not in VALID_ROLES:
            raise HTTPException(400, f"Rôle invalide. Valides: {VALID_ROLES}")
        updates.append(f"role = ${p}"); params.append(body.role); p += 1
    if body.is_active is not None:
        updates.append(f"is_active = ${p}"); params.append(body.is_active); p += 1
    if body.password is not None:
        try:
            pwd_hash = hash_password(body.password)
        except RuntimeError as e:
            raise HTTPException(500, str(e))
        updates.append(f"password_hash = ${p}"); params.append(pwd_hash); p += 1

    if not updates:
        raise HTTPException(400, "Aucune modification fournie")

    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE op_users SET {', '.join(updates)} WHERE id = ${p}",
                *params, user_id
            )
            if result == "UPDATE 0":
                raise HTTPException(404, "Utilisateur introuvable")

        await log_event(
            action=AuditAction.UPDATE_USER,
            user_id=int(caller["sub"]),
            user_email=caller["email"],
            resource=f"user:{user_id}",
            details={k: v for k, v in body.model_dump().items() if v is not None and k != "password"},
        )
        return {"id": user_id, "updated": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Admin/users] update {user_id}: {e}")
        raise HTTPException(500, "Erreur mise à jour utilisateur")


@router.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: int, request: Request, caller=Depends(_admin)):
    """Supprime un utilisateur. Interdit de se supprimer soi-même."""
    from api.database import get_pg_pool
    from core.audit  import log_event, AuditAction

    if user_id == int(caller["sub"]):
        raise HTTPException(400, "Impossible de supprimer votre propre compte")

    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            result = await conn.execute("DELETE FROM op_users WHERE id = $1", user_id)
            if result == "DELETE 0":
                raise HTTPException(404, "Utilisateur introuvable")

        await log_event(
            action=AuditAction.DELETE_USER,
            user_id=int(caller["sub"]),
            user_email=caller["email"],
            resource=f"user:{user_id}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Admin/users] delete {user_id}: {e}")
        raise HTTPException(500, "Erreur suppression utilisateur")


# ════════════════════════════════════════════════════════════
# PERMISSIONS SOURCES — CDC 2.6.1.B
# ════════════════════════════════════════════════════════════

@router.get("/users/{user_id}/permissions")
async def get_user_permissions(user_id: int, request: Request, _=Depends(_admin)):
    """Liste les sources autorisées pour un utilisateur."""
    from api.database import get_pg_pool

    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT usp.id, usp.source_id, usp.can_read,
                       usp.can_export, usp.can_query, usp.granted_at,
                       s.name AS source_name, s.connector_type
                FROM user_source_permissions usp
                LEFT JOIN sources s ON s.id::text = usp.source_id
                WHERE usp.user_id = $1
                ORDER BY usp.granted_at DESC
                """,
                user_id
            )
        return {
            "user_id": user_id,
            "permissions": [
                {
                    "id":           r["id"],
                    "source_id":    r["source_id"],
                    "source_name":  r["source_name"],
                    "connector_type": r["connector_type"],
                    "can_read":     r["can_read"],
                    "can_export":   r["can_export"],
                    "can_query":    r["can_query"],
                    "granted_at":   r["granted_at"].isoformat() if r["granted_at"] else None,
                }
                for r in rows
            ],
        }
    except Exception as e:
        logger.error(f"[Admin/perms] get {user_id}: {e}")
        raise HTTPException(500, "Erreur base de données")


@router.post("/users/{user_id}/permissions")
async def set_user_permission(
    user_id: int, body: SourcePermissionSet,
    request: Request, caller=Depends(_admin)
):
    """Accorde ou met à jour l'accès d'un utilisateur à une source."""
    from api.database import get_pg_pool
    from core.audit  import log_event, AuditAction

    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO user_source_permissions
                    (user_id, source_id, can_read, can_export, can_query, granted_by)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (user_id, source_id) DO UPDATE SET
                    can_read   = EXCLUDED.can_read,
                    can_export = EXCLUDED.can_export,
                    can_query  = EXCLUDED.can_query,
                    granted_by = EXCLUDED.granted_by
                """,
                user_id, body.source_id,
                body.can_read, body.can_export, body.can_query,
                int(caller["sub"])
            )

        await log_event(
            action=AuditAction.GRANT_PERM,
            user_id=int(caller["sub"]),
            user_email=caller["email"],
            resource=f"user:{user_id}:source:{body.source_id}",
            details=body.model_dump(),
        )
        return {"user_id": user_id, "source_id": body.source_id, "granted": True}

    except Exception as e:
        logger.error(f"[Admin/perms] set: {e}")
        raise HTTPException(500, "Erreur permission")


@router.delete("/users/{user_id}/permissions/{source_id}", status_code=204)
async def revoke_user_permission(
    user_id: int, source_id: str,
    request: Request, caller=Depends(_admin)
):
    """Révoque l'accès d'un utilisateur à une source."""
    from api.database import get_pg_pool
    from core.audit  import log_event, AuditAction

    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM user_source_permissions WHERE user_id=$1 AND source_id=$2",
                user_id, source_id
            )
        await log_event(
            action=AuditAction.REVOKE_PERM,
            user_id=int(caller["sub"]),
            user_email=caller["email"],
            resource=f"user:{user_id}:source:{source_id}",
        )
    except Exception as e:
        logger.error(f"[Admin/perms] revoke: {e}")
        raise HTTPException(500, "Erreur révocation")


# ════════════════════════════════════════════════════════════
# AUDIT LOGS — CDC 2.6.1 (Phase 10)
# ════════════════════════════════════════════════════════════

@router.get("/audit/logs")
async def get_audit_logs(
    request:    Request,
    limit:      int           = Query(50,  ge=1, le=500),
    offset:     int           = Query(0,   ge=0),
    user_email: Optional[str] = Query(None),
    action:     Optional[str] = Query(None),
    result:     Optional[str] = Query(None),
    date_from:  Optional[str] = Query(None),
    date_to:    Optional[str] = Query(None),
    _=Depends(_admin),
):
    """Retourne les audit logs avec filtres. Admin uniquement."""
    from core.audit import get_logs

    return await get_logs(
        limit=limit, offset=offset,
        user_email=user_email, action=action,
        result=result, date_from=date_from, date_to=date_to,
    )


@router.get("/audit/stats")
async def get_audit_stats(request: Request, _=Depends(_admin)):
    """Statistiques rapides pour le cockpit (24h)."""
    from core.audit import get_stats
    return await get_stats()


# ════════════════════════════════════════════════════════════
# CONFIG ADMIN — CDC 2.6.1.C (Voice) + 2.6.1.D (Dashboards)
# ════════════════════════════════════════════════════════════

@router.get("/config")
async def get_config(request: Request, _=Depends(_admin)):
    """Retourne toute la configuration admin."""
    from api.database import get_pg_pool
    import json

    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key, value, updated_at FROM admin_config ORDER BY key"
            )
        return {
            "config": {
                r["key"]: {
                    "value":      r["value"],
                    "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
                }
                for r in rows
            }
        }
    except Exception as e:
        logger.error(f"[Admin/config] get: {e}")
        raise HTTPException(500, "Erreur base de données")


@router.put("/config")
async def update_config(body: ConfigUpdate, request: Request, caller=Depends(_admin)):
    """Met à jour une clé de configuration."""
    from api.database import get_pg_pool
    from core.audit  import log_event, AuditAction
    import json

    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO admin_config (key, value, updated_by)
                VALUES ($1, $2::jsonb, $3)
                ON CONFLICT (key) DO UPDATE SET
                    value      = EXCLUDED.value,
                    updated_by = EXCLUDED.updated_by,
                    updated_at = NOW()
                """,
                body.key,
                json.dumps(body.value),
                int(caller["sub"])
            )

        await log_event(
            action=AuditAction.CONFIG_CHANGE,
            user_id=int(caller["sub"]),
            user_email=caller["email"],
            resource=f"config:{body.key}",
            details={"key": body.key, "value": body.value},
        )
        return {"key": body.key, "updated": True}

    except Exception as e:
        logger.error(f"[Admin/config] update: {e}")
        raise HTTPException(500, "Erreur mise à jour config")


# ════════════════════════════════════════════════════════════
# SOURCES STATUS — monitoring connecteurs pour cockpit
# ════════════════════════════════════════════════════════════

@router.get("/sources/status")
async def get_sources_status(request: Request, _=Depends(_admin)):
    """
    Retourne toutes les sources avec leur statut live.
    Utilisé par cockpit.html pour le monitoring connecteurs.
    """
    from api.database    import get_pg_pool
    from api.repository  import list_sources

    try:
        sources = await list_sources()
        result  = []
        for src in sources:
            result.append({
                "id":             str(src.id),
                "name":           src.name,
                "connector_type": src.connector_type,
                "status":         src.status or "unknown",
                "entity_count":   src.entity_count or 0,
                "test_latency_ms": src.test_latency_ms,
                "last_sync":      src.last_sync.isoformat() if src.last_sync else None,
                "error_count":    0,
            })
        return {"sources": result, "total": len(result)}
    except Exception as e:
        logger.error(f"[Admin/sources/status] {e}")
        raise HTTPException(500, "Erreur récupération sources")
