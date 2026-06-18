/**
 * op_auth.js — Module authentification partagé OnePilot
 * Chargé par : login.html, index.html, chat.html, admin.html
 * Dépend de : POST /auth/login, GET /auth/me/full
 * Phase 10 — Admin Console & Security
 */

const OpAuth = (() => {

  // ── Clés localStorage ──────────────────────────────────────
  const KEY_TOKEN   = 'op_token';
  const KEY_USER    = 'op_user';
  const KEY_ROLE    = 'op_role';
  const API         = '/api';

  // ── Lecture / écriture ─────────────────────────────────────

  function getToken() {
    return localStorage.getItem(KEY_TOKEN) || null;
  }

  function getUser() {
    try {
      const raw = localStorage.getItem(KEY_USER);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }

  function getRole() {
    return localStorage.getItem(KEY_ROLE) || null;
  }

  function isLoggedIn() {
    const token = getToken();
    if (!token) return false;
    // Vérifier expiration côté client (sans librairie jwt)
    try {
      const payload = JSON.parse(atob(token.split('.')[1]));
      return payload.exp * 1000 > Date.now();
    } catch {
      return false;
    }
  }

  function hasPermission(perm) {
    const user = getUser();
    if (!user) return false;
    return (user.permissions || []).includes(perm);
  }

  function canAccessSource(sourceId) {
    const user = getUser();
    if (!user) return false;
    const allowed = user.allowed_sources || [];
    // Liste vide = accès à tout (admin)
    if (allowed.length === 0) return true;
    return allowed.includes(sourceId);
  }

  // ── Auth Bearer header ─────────────────────────────────────

  function authHeaders() {
    const token = getToken();
    return token ? { 'Authorization': `Bearer ${token}` } : {};
  }

  async function apiFetch(path, opts = {}) {
    const headers = {
      'Content-Type': 'application/json',
      ...authHeaders(),
      ...(opts.headers || {}),
    };
    const res = await fetch(API + path, { ...opts, headers });
    if (res.status === 401) {
      // Token expiré ou invalide → logout
      _clearSession();
      window.location.href = '/login.html';
      return null;
    }
    return res;
  }

  // ── Session ────────────────────────────────────────────────

  function _saveSession(data) {
    localStorage.setItem(KEY_TOKEN, data.token);
    localStorage.setItem(KEY_ROLE,  data.role);
    localStorage.setItem(KEY_USER,  JSON.stringify({
      user_id:         data.user_id,
      email:           data.email,
      username:        data.username,
      role:            data.role,
      permissions:     data.permissions     || [],
      allowed_sources: data.allowed_sources || [],
    }));
  }

  function _clearSession() {
    localStorage.removeItem(KEY_TOKEN);
    localStorage.removeItem(KEY_USER);
    localStorage.removeItem(KEY_ROLE);
  }

  // ── Login ──────────────────────────────────────────────────

  async function login(email, password) {
    /**
     * Appelle POST /auth/login
     * Retourne { ok: true } ou { ok: false, error: string }
     */
    try {
      const res = await fetch(API + '/auth/login', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ email, password }),
      });

      if (res.ok) {
        const data = await res.json();
        _saveSession(data);
        return { ok: true, role: data.role };
      } else {
        const err = await res.json();
        return { ok: false, error: err.detail || 'Erreur de connexion' };
      }
    } catch (e) {
      return { ok: false, error: 'Serveur inaccessible' };
    }
  }

  // ── Logout ─────────────────────────────────────────────────

  async function logout() {
    /**
     * Notifie le backend (audit log) puis nettoie la session.
     */
    try {
      await apiFetch('/auth/logout', { method: 'POST' });
    } catch {}
    _clearSession();
    window.location.href = '/login.html';
  }

  // ── Guard ──────────────────────────────────────────────────

  function checkAuth(requiredRole = null) {
    /**
     * À appeler au DOMContentLoaded de chaque page protégée.
     * requiredRole : null = tout utilisateur connecté
     *               'admin' = admin uniquement
     *
     * Redirige vers login.html si non authentifié ou rôle insuffisant.
     */
    if (!isLoggedIn()) {
      _clearSession();
      window.location.href = '/login.html';
      return false;
    }

    if (requiredRole === 'admin' && getRole() !== 'admin') {
      // Rôle insuffisant → renvoyer vers chat.html (page user)
      window.location.href = '/chat.html';
      return false;
    }

    return true;
  }

  function redirectAfterLogin() {
    /**
     * Appelé par login.html après login réussi.
     * Admin → admin.html, autres → chat.html
     */
    const role = getRole();
    if (role === 'admin') {
      window.location.href = '/admin.html';
    } else {
      window.location.href = '/chat.html';
    }
  }

  // ── Injection UI ───────────────────────────────────────────

  function applyUserUI() {
    /**
     * Injecte nom/rôle/avatar dans les éléments existants de la page.
     * Cherche les IDs standards utilisés dans chat, admin, index.
     */
    const user = getUser();
    if (!user) return;

    const initial = (user.username || user.email || 'A')[0].toUpperCase();

    // Avatar / initiale
    ['sb-avatar', 'uAv', 'railUserAv'].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.textContent = initial;
        if (user.role === 'admin') el.classList.add('adm');
      }
    });

    // Nom
    ['sb-name', 'uName', 'hero-name'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = user.username || user.email;
    });

    // Rôle
    ['sb-role', 'uRole'].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        const labels = { admin: 'Admin Principal', power_user: 'Power User', user: 'Utilisateur' };
        el.textContent = labels[user.role] || user.role;
      }
    });

    // Éléments visibles admin uniquement
    ['sbAdminBtn', 'adminConsoleMenuItem', 'cdcMenuItem'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.style.display = user.role === 'admin' ? '' : 'none';
    });
  }

  // ── API publique ───────────────────────────────────────────

  return {
    // Lecture
    getToken,
    getUser,
    getRole,
    isLoggedIn,
    hasPermission,
    canAccessSource,
    // Auth
    login,
    logout,
    checkAuth,
    redirectAfterLogin,
    // HTTP
    authHeaders,
    apiFetch,
    // UI
    applyUserUI,
  };

})();
