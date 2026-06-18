"""
routes/auth.py — Endpoints authentification OnePilot
POST /auth/login  → vérifie credentials → retourne JWT
GET  /auth/me     → retourne profil utilisateur connecté
Phase 10 — Admin Console & Security
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Auth"])


# ── Schémas Pydantic ─────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email:    str
    password: str


class LoginResponse(BaseModel):
    token:    str
    user_id:  int
    email:    str
    username: str
    role:     str
    permissions:     List[str]
    allowed_sources: List[str]


class UserProfile(BaseModel):
    user_id:  int
    email:    str
    username: str
    role:     str
    permissions:     List[str]
    allowed_sources: List[str]


# ── POST /auth/login ─────────────────────────────────────────────────────────

@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest, request: Request):
    """
    Authentifie un utilisateur et retourne un JWT.
    Écrit un audit log LOGIN (success ou failure).
    """
    from api.database import get_pg_pool
    from core.auth  import verify_password, create_token
    from core.audit import log_event, AuditAction, AuditResult

    ip  = request.client.host if request.client else None
    ua  = request.headers.get("user-agent")

    # 1. Récupérer l'utilisateur par email
    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            user = await conn.fetchrow(
                "SELECT id, email, username, password_hash, role, is_active "
                "FROM op_users WHERE email = $1",
                req.email.lower().strip()
            )
    except Exception as e:
        logger.error(f"[Login] DB erreur: {e}")
        raise HTTPException(status_code=500, detail="Erreur base de données")

    # 2. Vérifications
    if not user:
        await log_event(
            action=AuditAction.LOGIN_FAILED,
            result=AuditResult.FAILURE,
            user_email=req.email,
            details={"reason": "email_not_found"},
            ip_address=ip, user_agent=ua,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou mot de passe incorrect"
        )

    if not user["is_active"]:
        await log_event(
            action=AuditAction.LOGIN_FAILED,
            result=AuditResult.FAILURE,
            user_id=user["id"], user_email=user["email"],
            details={"reason": "account_disabled"},
            ip_address=ip, user_agent=ua,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Compte désactivé"
        )

    if not verify_password(req.password, user["password_hash"]):
        await log_event(
            action=AuditAction.LOGIN_FAILED,
            result=AuditResult.FAILURE,
            user_id=user["id"], user_email=user["email"],
            details={"reason": "wrong_password"},
            ip_address=ip, user_agent=ua,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou mot de passe incorrect"
        )

    # 3. Récupérer les sources autorisées
    try:
        async with pool.acquire() as conn:
            perms = await conn.fetch(
                "SELECT source_id FROM user_source_permissions "
                "WHERE user_id = $1 AND can_read = TRUE",
                user["id"]
            )
            allowed_sources = [r["source_id"] for r in perms]
            # Mettre à jour last_login
            await conn.execute(
                "UPDATE op_users SET last_login = NOW() WHERE id = $1",
                user["id"]
            )
    except Exception as e:
        logger.warning(f"[Login] Erreur permissions: {e}")
        allowed_sources = []

    # 4. Générer le JWT
    from core.auth import ROLE_PERMISSIONS
    user_dict = {
        "id":              user["id"],
        "email":           user["email"],
        "username":        user["username"],
        "role":            user["role"],
        "allowed_sources": allowed_sources,
    }
    token = create_token(user_dict)

    # 5. Audit log succès
    await log_event(
        action=AuditAction.LOGIN,
        result=AuditResult.SUCCESS,
        user_id=user["id"], user_email=user["email"],
        details={"role": user["role"]},
        ip_address=ip, user_agent=ua,
    )

    logger.info(f"[Login] ✅ {user['email']} ({user['role']})")

    return LoginResponse(
        token=token,
        user_id=user["id"],
        email=user["email"],
        username=user["username"],
        role=user["role"],
        permissions=ROLE_PERMISSIONS.get(user["role"], []),
        allowed_sources=allowed_sources,
    )


# ── GET /auth/me ─────────────────────────────────────────────────────────────

@router.get("/me", response_model=UserProfile)
async def get_me(payload: Dict[str, Any] = Depends(lambda: None)):
    """
    Retourne le profil de l'utilisateur connecté depuis le JWT.
    """
    from core.auth import require_auth
    # Note: Depends(require_auth) est déclaré ici pour éviter l'import circulaire
    raise HTTPException(501, "Utilisez Depends(require_auth) directement")


@router.get("/me/full")
async def get_me_full(request: Request):
    """
    Retourne le profil complet depuis le JWT + données DB fraîches.
    """
    from core.auth  import extract_token_from_header, decode_token
    from api.database import get_pg_pool

    auth_header = request.headers.get("authorization")
    token = extract_token_from_header(auth_header)
    if not token:
        raise HTTPException(401, "Token manquant")

    payload = decode_token(token)
    if not payload:
        raise HTTPException(401, "Token invalide ou expiré")

    try:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            user = await conn.fetchrow(
                "SELECT id, email, username, role, is_active, last_login "
                "FROM op_users WHERE id = $1",
                int(payload["sub"])
            )
            if not user or not user["is_active"]:
                raise HTTPException(401, "Utilisateur inactif")

            perms = await conn.fetch(
                "SELECT source_id, can_read, can_export, can_query "
                "FROM user_source_permissions WHERE user_id = $1",
                user["id"]
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/auth/me] {e}")
        raise HTTPException(500, "Erreur base de données")

    from core.auth import ROLE_PERMISSIONS
    return {
        "user_id":        user["id"],
        "email":          user["email"],
        "username":       user["username"],
        "role":           user["role"],
        "last_login":     user["last_login"].isoformat() if user["last_login"] else None,
        "permissions":    ROLE_PERMISSIONS.get(user["role"], []),
        "source_permissions": [
            {
                "source_id":  p["source_id"],
                "can_read":   p["can_read"],
                "can_export": p["can_export"],
                "can_query":  p["can_query"],
            }
            for p in perms
        ],
    }


# ── POST /auth/logout ────────────────────────────────────────────────────────

@router.post("/logout")
async def logout(request: Request):
    """
    Logout côté serveur — écrit l'audit log.
    Le client doit supprimer le token de son localStorage.
    """
    from core.auth  import extract_token_from_header, decode_token
    from core.audit import log_event, AuditAction, AuditResult

    auth_header = request.headers.get("authorization")
    token   = extract_token_from_header(auth_header)
    payload = decode_token(token) if token else None
    ip      = request.client.host if request.client else None

    if payload:
        await log_event(
            action=AuditAction.LOGOUT,
            result=AuditResult.SUCCESS,
            user_id=int(payload.get("sub", 0)),
            user_email=payload.get("email"),
            ip_address=ip,
        )

    return {"message": "Déconnecté avec succès"}
