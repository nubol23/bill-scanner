const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000';

export function resolveApiBaseUrl(rawBaseUrl?: string) {
  const configuredBaseUrl = rawBaseUrl?.trim();
  if (configuredBaseUrl) {
    return configuredBaseUrl.replace(/\/$/, '');
  }

  if (import.meta.env.DEV) {
    return '';
  }

  return DEFAULT_API_BASE_URL;
}

export const API_BASE_URL = resolveApiBaseUrl(import.meta.env.VITE_API_BASE_URL);
export const APP_VERSION = import.meta.env.VITE_APP_VERSION?.trim() || '0.0.0';

export function buildApiUrl(path: string) {
  return `${API_BASE_URL}${path}`;
}
