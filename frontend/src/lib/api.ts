import { buildApiUrl } from './config';

export type RecognizeCandidate = {
  text: string;
  confidence: number;
};

export type RecognizeResponse = {
  status: 'ok' | 'not_found';
  serial: string | null;
  series: string | null;
  raw_text: string;
  confidence: number | null;
  candidates: RecognizeCandidate[];
  engine: string;
  latency_ms: number;
  request_id: string;
};

export type RecognizeClientContext = {
  deviceId: string;
  sessionId: string;
  pageLoadId: string;
  denomination: '10' | '20' | '50';
  torchEnabled: boolean;
  clientStartedAt: string;
};

export type FeedbackPayload = {
  deviceId: string;
  sessionId: string;
  pageLoadId: string;
  requestId: string | null;
  rating: 'up' | 'down';
  comment: string | null;
  promptedAfterScanCount: number;
};

export type ScanApiErrorKind =
  | 'timeout'
  | 'unavailable'
  | 'rate-limited'
  | 'invalid-image'
  | 'server';

export class ScanApiError extends Error {
  kind: ScanApiErrorKind;
  status: number | null;
  requestId: string | null;

  constructor(
    kind: ScanApiErrorKind,
    message: string,
    status: number | null = null,
    requestId: string | null = null,
  ) {
    super(message);
    this.name = 'ScanApiError';
    this.kind = kind;
    this.status = status;
    this.requestId = requestId;
  }
}

function buildRecognizeContextPayload(context: RecognizeClientContext) {
  return JSON.stringify({
    device_id: context.deviceId,
    session_id: context.sessionId,
    page_load_id: context.pageLoadId,
    denomination: context.denomination,
    torch_enabled: context.torchEnabled,
    client_started_at: context.clientStartedAt,
  });
}

function readRequestId(response: Response) {
  return response.headers.get('X-Request-Id');
}

export async function scanSerialImage(
  image: Blob,
  context: RecognizeClientContext,
  timeoutMs = 12_000,
): Promise<RecognizeResponse> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  const formData = new FormData();
  formData.append('image', image, 'serial-scan.jpg');
  formData.append('context', buildRecognizeContextPayload(context));

  try {
    const response = await fetch(buildApiUrl('/api/v1/recognize'), {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });
    const requestId = readRequestId(response);

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
        throw new ScanApiError('rate-limited', detail, response.status, requestId);
      }

      if ([400, 413, 415].includes(response.status)) {
        throw new ScanApiError('invalid-image', detail, response.status, requestId);
      }

      throw new ScanApiError('server', detail, response.status, requestId);
    }

    const recognizeResponse = payload as RecognizeResponse;
    return {
      ...recognizeResponse,
      request_id: requestId ?? recognizeResponse.request_id,
    };
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

export async function submitFeedback(payload: FeedbackPayload) {
  const response = await fetch(buildApiUrl('/api/v1/feedback'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      device_id: payload.deviceId,
      session_id: payload.sessionId,
      page_load_id: payload.pageLoadId,
      request_id: payload.requestId,
      rating: payload.rating,
      comment: payload.comment,
      prompted_after_scan_count: payload.promptedAfterScanCount,
    }),
  });

  if (!response.ok) {
    throw new Error('No se pudo registrar el comentario.');
  }
}
