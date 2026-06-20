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

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, status, UploadFile
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
# PLUGIN SYSTEM — CDC 2.6.1.A
# ════════════════════════════════════════════════════════════

@router.get("/plugins")
async def list_plugins(request: Request, _=Depends(_admin)):
    """Liste tous les connecteurs depuis DB + PluginManager."""
    from api.database import get_pg_pool
    try:
        from core.plugin_manager import plugin_manager
        pm_status = plugin_manager.status()
    except Exception:
        pm_status = {'registered': [], 'disabled': [], 'active_instances': []}
    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                'SELECT name, class_name, filename, file_path, enabled, uploaded_by, created_at '
                'FROM registered_plugins ORDER BY created_at DESC'
            )
        plugins = []
        for r in rows:
            in_memory = r['name'] in pm_status['registered']
            plugins.append({
                'name':             r['name'],
                'class_name':       r['class_name'],
                'module':           r['filename'],
                'enabled':          r['enabled'],
                'in_memory':        in_memory,
                'active_instances': sum(1 for k in pm_status['active_instances'] if k.startswith(r['name'])),
                'uploaded_by':      r['uploaded_by'],
                'created_at':       r['created_at'].isoformat() if r['created_at'] else None,
            })
        return {'plugins': plugins, 'total': len(plugins)}
    except Exception as e:
        logger.error(f'[Admin/plugins] list: {e}')
        return {'plugins': [], 'total': 0}


@router.post("/plugins/{plugin_name}/toggle")
async def toggle_plugin(
    plugin_name: str,
    request: Request,
    caller=Depends(_admin),
):
    """Active ou désactive un connecteur custom."""
    from core.plugin_manager import plugin_manager
    from core.audit import log_event, AuditAction
    if plugin_name not in plugin_manager.list_registered():
        raise HTTPException(404, f"Plugin '{plugin_name}' non trouvé")
    currently_enabled = plugin_manager.is_enabled(plugin_name)
    if currently_enabled:
        plugin_manager.disable(plugin_name)
        new_state = 'disabled'
    else:
        plugin_manager.enable(plugin_name)
        new_state = 'enabled'
    await log_event(
        action=AuditAction.CONFIG_CHANGE,
        user_id=int(caller['sub']), user_email=caller['email'],
        resource=f'plugin:{plugin_name}',
        details={'action': 'toggle', 'new_state': new_state},
    )
    # Persister l'état en DB
    try:
        from api.database import get_pg_pool
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                'UPDATE registered_plugins SET enabled=$1, updated_at=NOW() WHERE name=$2',
                new_state == 'enabled', plugin_name
            )
    except Exception as e:
        logger.warning(f'[Plugins] Erreur update DB: {e}')
    return {'plugin': plugin_name, 'enabled': new_state == 'enabled'}


@router.post("/plugins/upload", status_code=201)
async def upload_plugin(
    request: Request,
    file: UploadFile = File(...),
    caller=Depends(_admin),
):
    """
    Upload d'un connecteur custom .py.
    Valide que le fichier contient une classe héritant de BaseConnector
    avec les 4 méthodes abstraites requises.
    """
    import ast, os, importlib.util
    from core.audit import log_event, AuditAction

    if not file.filename.endswith('.py'):
        raise HTTPException(400, 'Le fichier doit être un fichier Python (.py)')

    content_bytes = await file.read()
    if len(content_bytes) > 500_000:
        raise HTTPException(400, 'Fichier trop volumineux (max 500KB)')

    source_code = content_bytes.decode('utf-8', errors='replace')

    # ── Validation AST ──
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        raise HTTPException(400, f'Erreur de syntaxe Python: {e}')

    # Chercher une classe héritant de BaseConnector
    required_methods = {'connect', 'test_connection', 'get_metadata', 'execute_query'}
    found_class = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef): continue
        bases = [b.id if isinstance(b, ast.Name) else
                 b.attr if isinstance(b, ast.Attribute) else ''
                 for b in node.bases]
        if 'BaseConnector' not in bases: continue
        methods = {n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)}
        missing = required_methods - methods
        if missing:
            raise HTTPException(400,
                f"Classe '{node.name}': méthodes manquantes: {', '.join(missing)}")
        found_class = node.name
        break

    if not found_class:
        raise HTTPException(400,
            'Aucune classe héritant de BaseConnector trouvée. '
            'Votre classe doit étendre BaseConnector.')

    # ── Sauvegarder dans /app/api/connectors/custom/ ──
    plugin_dir = '/app/api/connectors/custom'
    os.makedirs(plugin_dir, exist_ok=True)
    plugin_name = file.filename[:-3]  # sans .py
    plugin_path = os.path.join(plugin_dir, file.filename)
    with open(plugin_path, 'w', encoding='utf-8') as f:
        f.write(source_code)

    # ── Enregistrer dans le PluginManager ──
    try:
        from core.plugin_manager import plugin_manager
        spec   = importlib.util.spec_from_file_location(plugin_name, plugin_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls    = getattr(module, found_class)
        plugin_manager.register(plugin_name, cls)
    except Exception as e:
        os.remove(plugin_path)
        raise HTTPException(500, f'Erreur chargement plugin: {e}')

    # ── Persister en DB ──
    try:
        from api.database import get_pg_pool
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO registered_plugins (name, class_name, filename, file_path, enabled, uploaded_by)
                VALUES ($1, $2, $3, $4, TRUE, $5)
                ON CONFLICT (name) DO UPDATE SET
                    class_name  = EXCLUDED.class_name,
                    filename    = EXCLUDED.filename,
                    file_path   = EXCLUDED.file_path,
                    enabled     = TRUE,
                    updated_at  = NOW()
            """, plugin_name, found_class, file.filename, plugin_path,
                 caller.get('email', 'admin'))
    except Exception as e:
        logger.warning(f'[Plugins] Erreur persistance DB: {e}')
        # Non bloquant — le plugin est quand même enregistré en mémoire

    await log_event(
        action=AuditAction.CONFIG_CHANGE,
        user_id=int(caller['sub']), user_email=caller['email'],
        resource=f'plugin:{plugin_name}',
        details={'action': 'upload', 'class': found_class, 'file': file.filename},
    )

    return {
        'plugin_name':  plugin_name,
        'class_name':   found_class,
        'filename':     file.filename,
        'registered':   True,
        'message':      f"Plugin '{plugin_name}' enregistré avec succès",
    }


# ════════════════════════════════════════════════════════════
# SECRETS — Chiffrement AES-256-GCM
# ════════════════════════════════════════════════════════════

@router.post("/secrets/rotate")
async def rotate_secrets(request: Request, caller=Depends(_admin)):
    """
    Rechiffre tous les secrets en clair avec AES-256-GCM.
    CDC §2.1.7 : Rotation automatique des credentials.
    """
    from api.database import get_pg_pool
    from core.audit   import log_event, AuditAction
    try:
        from core.secrets import rotate_all_secrets
        pool   = await get_pg_pool()
        result = await rotate_all_secrets(pool)
        await log_event(
            action=AuditAction.CONFIG_CHANGE,
            user_id=int(caller['sub']), user_email=caller['email'],
            resource='connection_secrets',
            details={'action': 'rotate_all', **result},
        )
        return {
            'success':   True,
            'encrypted': result['encrypted'],
            'skipped':   result['skipped'],
            'message':   f"{result['encrypted']} secret(s) chiffré(s), {result['skipped']} déjà chiffré(s)"
        }
    except Exception as e:
        logger.error(f'[Admin/secrets/rotate] {e}')
        raise HTTPException(500, f'Erreur rotation: {e}')


@router.post("/secrets/rotate/{source_id}")
async def rotate_source_secret(source_id: str, request: Request, caller=Depends(_admin)):
    """Chiffre les credentials d'une seule source."""
    from api.database import get_pg_pool
    from core.audit   import log_event, AuditAction
    try:
        from core.secrets import encrypt_secret, is_encrypted
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, secret_value FROM connection_secrets WHERE source_id=$1",
                source_id
            )
            if not rows:
                raise HTTPException(404, 'Aucun credential pour cette source')
            count = 0
            for row in rows:
                val = row['secret_value']
                if val and not is_encrypted(val):
                    new_val = encrypt_secret(val)
                    await conn.execute(
                        'UPDATE connection_secrets SET secret_value=$1 WHERE id=$2',
                        new_val, row['id']
                    )
                    count += 1
        await log_event(
            action=AuditAction.CONFIG_CHANGE,
            user_id=int(caller['sub']), user_email=caller['email'],
            resource=f'secret:{source_id}',
            details={'action': 'rotate_single', 'encrypted': count},
        )
        return {'success': True, 'source_id': source_id, 'encrypted': count,
                'message': f'{count} credential(s) chiffré(s)'}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'[Admin/secrets/rotate/{source_id}] {e}')
        raise HTTPException(500, f'Erreur: {e}')


@router.get("/secrets/status")
async def secrets_status(request: Request, _=Depends(_admin)):
    """Retourne le statut de chiffrement de chaque source."""
    from api.database import get_pg_pool
    try:
        from core.secrets import is_encrypted
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT cs.source_id, ds.name, ds.connector_type, "
                "       cs.secret_key, cs.secret_value "
                "FROM connection_secrets cs "
                "JOIN data_sources ds ON ds.id = cs.source_id"
            )
        result = []
        for r in rows:
            ct = r['connector_type']
            ct_str = ct.value if hasattr(ct, 'value') else str(ct)
            result.append({
                'source_id':     str(r['source_id']),
                'source_name':   r['name'],
                'connector_type':ct_str,
                'secret_key':    r['secret_key'],
                'encrypted':     is_encrypted(r['secret_value'] or ''),
            })
        total      = len(result)
        encrypted  = sum(1 for x in result if x['encrypted'])
        return {
            'secrets':        result,
            'total':          total,
            'encrypted':      encrypted,
            'not_encrypted':  total - encrypted,
        }
    except Exception as e:
        logger.error(f'[Admin/secrets/status] {e}')
        raise HTTPException(500, 'Erreur statut secrets')


# ════════════════════════════════════════════════════════════
# MONITORING CONNECTEURS
# ════════════════════════════════════════════════════════════

@router.get("/sources/metrics")
async def get_sources_metrics(request: Request, window_hours: int = 24, _=Depends(_admin)):
    from api.database   import get_pg_pool
    from api.repository import list_sources
    try:
        pool    = await get_pg_pool()
        sources = await list_sources()
        result  = []
        async with pool.acquire() as conn:
            for src in sources:
                src_id = str(src.id)
                ct     = src.connector_type
                ct_str = ct.value if hasattr(ct, 'value') else str(ct)
                row = await conn.fetchrow(
                    "SELECT "
                    "  COUNT(*) FILTER (WHERE action='QUERY') AS query_count, "
                    "  COUNT(*) FILTER (WHERE action='QUERY' AND result='failure') AS error_count, "
                    "  COUNT(DISTINCT user_email) AS unique_users, "
                    "  MAX(created_at) AS last_activity "
                    "FROM audit_logs "
                    "WHERE resource LIKE $1 "
                    "  AND created_at >= NOW() - ($2 || ' hours')::INTERVAL",
                    f"%{src_id}%", str(window_hours)
                )
                total      = int(row['query_count'] or 0)
                errors     = int(row['error_count']  or 0)
                error_rate = round(errors / total * 100, 1) if total > 0 else 0.0
                cb_state   = 'unknown'
                try:
                    from api.connection_service import CircuitBreaker
                    cb       = CircuitBreaker.get(src_id, src.options or {})
                    cb_state = cb.state.value
                except Exception:
                    pass
                result.append({
                    'id':              src_id,
                    'name':            src.name,
                    'connector_type':  ct_str,
                    'status':          src.status or 'unknown',
                    'entity_count':    src.entity_count or 0,
                    'test_latency_ms': src.test_latency_ms,
                    'last_synced_at':  src.last_synced_at.isoformat() if src.last_synced_at else None,
                    'last_tested_at':  src.last_tested_at.isoformat() if src.last_tested_at else None,
                    'query_count':     total,
                    'error_count':     errors,
                    'error_rate_pct':  error_rate,
                    'unique_users':    int(row['unique_users'] or 0),
                    'last_activity':   row['last_activity'].isoformat() if row['last_activity'] else None,
                    'cb_state':        cb_state,
                })
        result.sort(key=lambda x: (-int(x['status'] in ['error','disconnected']), -x['query_count']))
        return {'sources': result, 'total': len(result), 'window_hours': window_hours}
    except Exception as e:
        logger.error(f'[Admin/sources/metrics] {e}')
        raise HTTPException(500, 'Erreur métriques sources')


# ════════════════════════════════════════════════════════════
# ACTIONS UTILISATEUR — Reset password · Toggle statut · Stats
# ════════════════════════════════════════════════════════════

@router.post("/users/{user_id}/reset-password")
async def reset_password(user_id: int, request: Request, caller=Depends(_admin)):
    """Génère un lien de reset valable 24h. L'admin copie le lien manuellement."""
    import uuid
    from api.database import get_pg_pool
    from core.audit   import log_event, AuditAction
    token = str(uuid.uuid4())
    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            user = await conn.fetchrow(
                "SELECT id, email, username FROM op_users WHERE id = $1", user_id
            )
            if not user:
                raise HTTPException(404, "Utilisateur introuvable")
            sql = (
                "INSERT INTO password_reset_tokens (user_id, token, expires_at) "
                "VALUES ($1, $2, NOW() + INTERVAL '24 hours') "
                "ON CONFLICT (user_id) DO UPDATE "
                "    SET token = EXCLUDED.token, "
                "        expires_at = EXCLUDED.expires_at, "
                "        created_at = NOW()"
            )
            await conn.execute(sql, user_id, token)
        await log_event(
            action=AuditAction.CONFIG_CHANGE,
            user_id=int(caller["sub"]), user_email=caller["email"],
            resource=f"user:{user_id}",
            details={"action": "reset_password_link", "target": user["email"]},
        )
        base = str(request.base_url).rstrip("/")
        return {"reset_url": f"{base}/reset-password?token={token}",
                "token": token, "expires_in": "24h", "user_email": user["email"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Admin/users] reset-password {user_id}: {e}")
        raise HTTPException(500, "Erreur génération token reset")


@router.post("/users/{user_id}/toggle-status")
async def toggle_user_status(user_id: int, request: Request, caller=Depends(_admin)):
    """Active ou désactive un compte utilisateur."""
    from api.database import get_pg_pool
    from core.audit   import log_event, AuditAction
    if user_id == int(caller["sub"]):
        raise HTTPException(400, "Impossible de modifier votre propre statut")
    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            user = await conn.fetchrow(
                "SELECT id, email, is_active FROM op_users WHERE id = $1", user_id
            )
            if not user:
                raise HTTPException(404, "Utilisateur introuvable")
            new_status = not user["is_active"]
            await conn.execute(
                "UPDATE op_users SET is_active = $1, updated_at = NOW() WHERE id = $2",
                new_status, user_id
            )
        await log_event(
            action=AuditAction.UPDATE_USER,
            user_id=int(caller["sub"]), user_email=caller["email"],
            resource=f"user:{user_id}",
            details={"action": "toggle_status", "new_status": new_status},
        )
        return {"user_id": user_id, "is_active": new_status}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Admin/users] toggle-status {user_id}: {e}")
        raise HTTPException(500, "Erreur modification statut")


@router.get("/users/{user_id}/stats")
async def get_user_stats(user_id: int, request: Request, _=Depends(_admin)):
    """Statistiques d'activité depuis audit_logs."""
    from api.database import get_pg_pool
    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            user = await conn.fetchrow(
                "SELECT email FROM op_users WHERE id = $1", user_id
            )
            if not user:
                raise HTTPException(404, "Utilisateur introuvable")
            row = await conn.fetchrow(
                "SELECT "
                "  COUNT(*) FILTER (WHERE action='QUERY') AS query_count, "
                "  COUNT(*) FILTER (WHERE action='LOGIN') AS login_count, "
                "  COUNT(*) FILTER (WHERE action='QUERY' "
                "    AND created_at >= NOW() - INTERVAL '7 days') AS queries_7d, "
                "  MAX(created_at) FILTER (WHERE action='QUERY') AS last_query_at "
                "FROM audit_logs WHERE user_email = $1",
                user["email"]
            )
            return {
                "user_id":       user_id,
                "query_count":   int(row["query_count"]  or 0),
                "login_count":   int(row["login_count"]  or 0),
                "queries_7d":    int(row["queries_7d"]   or 0),
                "last_query_at": row["last_query_at"].isoformat() if row["last_query_at"] else None,
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Admin/users] stats {user_id}: {e}")
        return {"user_id": user_id, "query_count": 0, "login_count": 0,
                "queries_7d": 0, "last_query_at": None}


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
                LEFT JOIN data_sources s ON s.id::text = usp.source_id
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

@router.get("/audit/anomalies")
async def detect_anomalies(request: Request, _=Depends(_admin)):
    from api.database import get_pg_pool
    from datetime import datetime, timezone
    try:
        pool = await get_pg_pool()
        anomalies = []
        async with pool.acquire() as conn:
            # Brute force : > 5 LOGIN_FAILED depuis la même IP en 1h
            rows = await conn.fetch(
                "SELECT ip_address, COUNT(*) AS cnt FROM audit_logs "
                "WHERE action='LOGIN_FAILED' AND created_at >= NOW() - INTERVAL '1 hour' "
                "GROUP BY ip_address HAVING COUNT(*) >= 5"
            )
            for r in rows:
                anomalies.append({
                    'type': 'brute_force', 'severity': 'high',
                    'message': f"IP {r['ip_address']}: {r['cnt']} tentatives échouées en 1h",
                })
            # Volume anormal : > 50 QUERY en 1h pour un user
            rows2 = await conn.fetch(
                "SELECT user_email, COUNT(*) AS cnt FROM audit_logs "
                "WHERE action='QUERY' AND created_at >= NOW() - INTERVAL '1 hour' "
                "GROUP BY user_email HAVING COUNT(*) >= 50"
            )
            for r in rows2:
                anomalies.append({
                    'type': 'high_volume', 'severity': 'medium',
                    'message': f"{r['user_email']}: {r['cnt']} requêtes en 1h (volume anormal)",
                })
            # Accès refusés répétés
            rows3 = await conn.fetch(
                "SELECT user_email, COUNT(*) AS cnt FROM audit_logs "
                "WHERE action='ACCESS_DENIED' AND created_at >= NOW() - INTERVAL '24 hours' "
                "GROUP BY user_email HAVING COUNT(*) >= 3"
            )
            for r in rows3:
                anomalies.append({
                    'type': 'access_denied', 'severity': 'medium',
                    'message': f"{r['user_email']}: {r['cnt']} accès refusés en 24h",
                })
        return {'anomalies': anomalies, 'checked_at': datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        logger.error(f'[Admin/audit/anomalies] {e}')
        return {'anomalies': [], 'checked_at': ''}


@router.get("/audit/logs")
async def get_audit_logs(
    request:    Request,
    limit:      int           = Query(50,  ge=1, le=1000),
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
    from core.audit import get_stats
    from api.database import get_pg_pool
    stats = await get_stats()
    # Ajouter queries_today
    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            queries = await conn.fetchval(
                "SELECT COUNT(*) FROM audit_logs "
                "WHERE action='QUERY' AND created_at >= NOW() - INTERVAL '24 hours'"
            )
        stats['queries_today'] = int(queries or 0)
    except Exception:
        stats['queries_today'] = 0
    return stats

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
                "last_sync":      src.last_synced_at.isoformat() if src.last_synced_at else None,
                "error_count":    0,
            })
        return {"sources": result, "total": len(result)}
    except Exception as e:
        logger.error(f"[Admin/sources/status] {e}")
        raise HTTPException(500, "Erreur récupération sources")
