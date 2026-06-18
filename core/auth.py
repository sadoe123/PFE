"""
core/auth.py — Authentification utilisateurs OnePilot
JWT HS256 · bcrypt · indépendant de core/auth_manager.py (qui gère les connecteurs ERP)
Phase 10 — Admin Console & Security
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# ── Constantes ──────────────────────────────────────────────────────────────
JWT_SECRET  = os.environ.get("JWT_SECRET", "onepilot_jwt_secret_change_in_prod")
JWT_ALGO    = "HS256"
JWT_EXPIRE  = int(os.environ.get("JWT_EXPIRE_HOURS", "24"))

# Permissions par rôle — correspond à CDC 2.6.1.B
ROLE_PERMISSIONS: Dict[str, list] = {
    "admin": [
        "manage_connectors",
        "manage_users",
        "view_all_data",
        "validate_responses",
        "create_dashboards",
        "export_data",
        "view_audit_logs",
        "configure_voice",
        "configure_dashboards",
    ],
    "power_user": [
        "create_dashboards",
        "export_data",
        "view_assigned_sources",
    ],
    "user": [
        "ask_questions",
        "view_own_history",
    ],
}

# ── Dépendances optionnelles ─────────────────────────────────────────────────
try:
    import bcrypt as _bcrypt
    _BCRYPT_OK = True
except ImportError:
    _BCRYPT_OK = False
    logger.warning("[Auth] bcrypt non disponible — hash password désactivé")

try:
    import jwt as _jwt
    _JWT_OK = True
except ImportError:
    _JWT_OK = False
    logger.warning("[Auth] PyJWT non disponible — JWT désactivé")


# ── Password ─────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Hash un mot de passe avec bcrypt. Lève RuntimeError si bcrypt absent."""
    if not _BCRYPT_OK:
        raise RuntimeError("bcrypt non disponible — installez : pip install bcrypt")
    return _bcrypt.hashpw(password.encode("utf-8"), _bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Vérifie un mot de passe contre son hash bcrypt."""
    if not _BCRYPT_OK:
        return False
    try:
        return _bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception as e:
        logger.warning(f"[Auth] verify_password erreur: {e}")
        return False


# ── JWT ──────────────────────────────────────────────────────────────────────

def create_token(user: Dict[str, Any], expire_hours: int = JWT_EXPIRE) -> str:
    """
    Génère un JWT signé HS256.
    Payload : user_id, email, username, role, permissions, allowed_sources, exp
    """
    if not _JWT_OK:
        raise RuntimeError("PyJWT non disponible — installez : pip install PyJWT")

    now = datetime.now(timezone.utc)
    payload = {
        "sub":             str(user["id"]),
        "email":           user["email"],
        "username":        user["username"],
        "role":            user["role"],
        "permissions":     ROLE_PERMISSIONS.get(user["role"], []),
        "allowed_sources": user.get("allowed_sources", []),   # [] = accès à tout
        "iat":             now,
        "exp":             now + timedelta(hours=expire_hours),
    }
    return _jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Décode et vérifie un JWT.
    Retourne le payload ou None si invalide/expiré.
    """
    if not _JWT_OK:
        return None
    try:
        return _jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except _jwt.ExpiredSignatureError:
        logger.debug("[Auth] Token expiré")
        return None
    except _jwt.InvalidTokenError as e:
        logger.debug(f"[Auth] Token invalide: {e}")
        return None


def extract_token_from_header(authorization: Optional[str]) -> Optional[str]:
    """Extrait le token depuis 'Authorization: Bearer <token>'."""
    if not authorization:
        return None
    parts = authorization.strip().split(" ")
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


# ── FastAPI Dependency ────────────────────────────────────────────────────────

from fastapi import Header, HTTPException, status


async def require_auth(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """
    Dépendance FastAPI — vérifie le JWT dans le header Authorization.
    Usage : user = Depends(require_auth)
    """
    token = extract_token_from_header(authorization)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token manquant",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


async def require_admin(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """
    Dépendance FastAPI — vérifie JWT + rôle admin.
    Usage : user = Depends(require_admin)
    """
    payload = await require_auth(authorization)
    if payload.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès réservé aux administrateurs",
        )
    return payload


def check_source_access(payload: Dict[str, Any], source_id: str) -> bool:
    """
    Vérifie si l'utilisateur a accès à une source donnée.
    Si allowed_sources est vide → accès à tout (admin).
    """
    allowed = payload.get("allowed_sources", [])
    if not allowed:
        return True
    return source_id in allowed
