const STORAGE_KEYS = {
  access: "spotify_access_token",
  refresh: "spotify_refresh_token",
  expiresAt: "spotify_expires_at",
};

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:5174";

const getTokensFromUrl = () => {
  const params = new URLSearchParams(window.location.search);
  const access_token = params.get("access_token");
  if (!access_token) {
    return null;
  }

  return {
    access_token,
    refresh_token: params.get("refresh_token"),
    expires_in: params.get("expires_in"),
  };
};

const saveTokens = ({ access_token, refresh_token, expires_in }) => {
  const accessToken = access_token || "";
  const refreshToken =
    refresh_token || localStorage.getItem(STORAGE_KEYS.refresh) || "";
  const expiresIn = Number(expires_in || 3600);
  const expiresAt = Date.now() + expiresIn * 1000;

  if (accessToken) {
    localStorage.setItem(STORAGE_KEYS.access, accessToken);
  }
  if (refreshToken) {
    localStorage.setItem(STORAGE_KEYS.refresh, refreshToken);
  }
  if (expiresAt) {
    localStorage.setItem(STORAGE_KEYS.expiresAt, String(expiresAt));
  }

  return { accessToken, refreshToken, expiresAt };
};

const loadTokens = () => {
  return {
    accessToken: localStorage.getItem(STORAGE_KEYS.access) || "",
    refreshToken: localStorage.getItem(STORAGE_KEYS.refresh) || "",
    expiresAt: Number(localStorage.getItem(STORAGE_KEYS.expiresAt) || 0),
  };
};

const clearTokens = () => {
  localStorage.removeItem(STORAGE_KEYS.access);
  localStorage.removeItem(STORAGE_KEYS.refresh);
  localStorage.removeItem(STORAGE_KEYS.expiresAt);
};

export {
  API_BASE,
  STORAGE_KEYS,
  getTokensFromUrl,
  saveTokens,
  loadTokens,
  clearTokens,
};
