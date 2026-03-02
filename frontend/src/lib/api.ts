export type RecognizeCandidate = {
  text: string;
  confidence: number;
};

export type RecognizeResponse = {
  status: 'ok' | 'not_found';
  serial: string | null;
  raw_text: string;
  confidence: number | null;
  candidates: RecognizeCandidate[];
  engine: string;
  latency_ms: number;
  request_id: string;
};

export type ScanApiErrorKind =
  | 'timeout'
  | 'unavailable'
  | 'rate-limited'
  | 'invalid-image'
  | 'server';

const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000';
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE_URL).replace(/\/$/, '');

export class ScanApiError extends Error {
  kind: ScanApiErrorKind;
  status: number | null;

  constructor(kind: ScanApiErrorKind, message: string, status: number | null = null) {
    super(message);
    this.name = 'ScanApiError';
    this.kind = kind;
    this.status = status;
  }
}

export async function scanSerialImage(
  image: Blob,
  timeoutMs = 12_000,
): Promise<RecognizeResponse> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  const formData = new FormData();
  formData.append('image', image, 'serial-scan.jpg');

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/recognize`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    let payload: unknown = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }

    if (!response.ok) {
      const detail =
        typeof payload === 'object' && payload && 'detail' in payload
          ? String((payload as { detail: unknown }).detail)
          : 'No se pudo procesar la imagen.';

      if (response.status === 429) {
        throw new ScanApiError('rate-limited', detail, response.status);
      }

      if ([400, 413, 415].includes(response.status)) {
        throw new ScanApiError('invalid-image', detail, response.status);
      }

      throw new ScanApiError('server', detail, response.status);
    }

    return payload as RecognizeResponse;
  } catch (error) {
    if (error instanceof ScanApiError) {
      throw error;
    }

    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ScanApiError(
        'timeout',
        'El servidor tardó demasiado en responder. Intenta de nuevo.',
      );
    }

    throw new ScanApiError(
      'unavailable',
      'No se pudo conectar con el servicio de análisis. Verifica tu conexión e intenta de nuevo.',
    );
  } finally {
    window.clearTimeout(timeoutId);
  }
}
