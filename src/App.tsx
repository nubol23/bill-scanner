import { useEffect, useRef, useState } from 'react';
import { Camera, CheckCircle2, ImageUp, Loader2, XCircle } from 'lucide-react';

import {
  ensurePaddleOcrReady,
  looksLikePaddleMemoryError,
  PaddleOcrTimeoutError,
  scanWithPaddleOcr,
  type DetectedSerial,
  type ScanFeedback,
  type SerialSource,
} from './lib/paddle-ocr';

const CAMERA_FRAME_WIDTH_RATIO = 0.78;
const CAMERA_FRAME_HEIGHT_RATIO = 0.18;
const DERIVATIVE_JPEG_QUALITY = 0.92;
const FOREGROUND_ANALYSIS_MAX_SIDE = 256;
const FOREGROUND_COLOR_DISTANCE_THRESHOLD = 44;
const FOREGROUND_CHROMA_THRESHOLD = 20;

type RelativeCropPreset = {
  id: string;
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
};

type CropRect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type UploadCropCandidate = {
  id: string;
  label: string;
  rect: CropRect;
};

const UPLOAD_SCAN_PRESETS: readonly RelativeCropPreset[] = [
  {
    id: 'top-right-left',
    label: 'serial superior derecho',
    x: 0.44,
    y: 0,
    width: 0.42,
    height: 0.08,
  },
  {
    id: 'top-right-right',
    label: 'serial superior derecho desplazado',
    x: 0.54,
    y: 0,
    width: 0.42,
    height: 0.08,
  },
  {
    id: 'bottom-left-left',
    label: 'serial inferior izquierdo',
    x: 0,
    y: 0.84,
    width: 0.42,
    height: 0.08,
  },
  {
    id: 'bottom-left-right',
    label: 'serial inferior izquierdo desplazado',
    x: 0.08,
    y: 0.84,
    width: 0.42,
    height: 0.08,
  },
  {
    id: 'top-right-band',
    label: 'banda superior derecha',
    x: 0.4,
    y: 0,
    width: 0.56,
    height: 0.1,
  },
  {
    id: 'bottom-left-band',
    label: 'banda inferior izquierda',
    x: 0,
    y: 0.82,
    width: 0.56,
    height: 0.1,
  },
] as const;

const validRanges = [
  [67250001, 67700000],
  [69050001, 69500000],
  [69500001, 69950000],
  [69950001, 70400000],
  [70400001, 70850000],
  [70850001, 71300000],
  [76310012, 85139995],
  [86400001, 86850000],
  [90900001, 91350000],
  [91800001, 92250000],
  [87280145, 91646549],
  [96650001, 97100000],
  [99800001, 100250000],
  [100250001, 100700000],
  [109250001, 109700000],
  [110600001, 111050000],
  [111050001, 111500000],
  [111950001, 112400000],
  [112400001, 112850000],
  [112850001, 113300000],
  [114200001, 114650000],
  [114650001, 115100000],
  [115100001, 115550000],
  [118700001, 119150000],
  [119150001, 119600000],
  [120500001, 120950000],
  [77100001, 77550000],
  [78000001, 78450000],
  [78900001, 96350000],
  [96350001, 96800000],
  [96800001, 97250000],
  [98150001, 98600000],
  [104900001, 105350000],
  [105350001, 105800000],
  [106700001, 107150000],
  [107600001, 108050000],
  [108050001, 108500000],
  [109400001, 109850000],
] as const;

type OcrStatus = 'idle' | 'initializing' | 'ready' | 'error';

type TorchSettings = {
  torch?: boolean;
};

type TorchCapableTrack = MediaStreamTrack & {
  getCapabilities?: () => MediaTrackCapabilities & TorchSettings;
};

type SerialResult = {
  isValid: boolean;
  serialDisplay: string;
  serialNumeric: number;
  source: SerialSource;
  confidence: number;
};

type LoadingProgressState = {
  progress: number | null;
  label: string;
  detail?: string;
};

type CandidateVote = {
  confidenceTotal: number;
  hits: number;
  maxConfidence: number;
  source: SerialSource;
};

type AggregatedScanResult = {
  serials: DetectedSerial[];
  feedback: ScanFeedback;
};

type DiagnosticResult = {
  fileName: string;
  expected: string | null;
  detected: string | null;
  confidence: number | null;
  feedback: ScanFeedback;
  matched: boolean | null;
  error: string | null;
};

const noop = () => undefined;

function clampProgress(value: number) {
  return Math.max(0, Math.min(100, value));
}

function clampCropValue(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function LoadingProgressBar({ progress }: { progress: LoadingProgressState | null }) {
  if (!progress) {
    return null;
  }

  const normalizedProgress =
    typeof progress.progress === 'number' ? clampProgress(progress.progress) : null;

  return (
    <div className="progress-panel" role="status" aria-live="polite">
      <div className="progress-meta">
        <span className="progress-label">{progress.label}</span>
        {normalizedProgress !== null && (
          <span className="progress-value">{Math.round(normalizedProgress)}%</span>
        )}
      </div>
      {normalizedProgress !== null && (
        <div
          className="progress-track"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={normalizedProgress}
          aria-valuetext={`${Math.round(normalizedProgress)}%`}
        >
          <div className="progress-fill" style={{ width: `${normalizedProgress}%` }} />
        </div>
      )}
      {progress.detail && <small className="progress-detail">{progress.detail}</small>}
    </div>
  );
}

function waitForNextPaint(frames = 1) {
  return new Promise<void>((resolve) => {
    const step = (remainingFrames: number) => {
      if (remainingFrames <= 0) {
        resolve();
        return;
      }

      window.requestAnimationFrame(() => step(remainingFrames - 1));
    };

    step(frames);
  });
}

function getCaptureGuideRect(width: number, height: number) {
  const guideWidth = Math.max(1, Math.round(width * CAMERA_FRAME_WIDTH_RATIO));
  const guideHeight = Math.max(1, Math.round(height * CAMERA_FRAME_HEIGHT_RATIO));

  return {
    x: Math.max(0, Math.round((width - guideWidth) / 2)),
    y: Math.max(0, Math.round((height - guideHeight) / 2)),
    width: guideWidth,
    height: guideHeight,
  };
}

function resolveRelativeCropRect(
  width: number,
  height: number,
  preset: RelativeCropPreset,
  baseRect?: CropRect,
): CropRect {
  const area = baseRect ?? {
    x: 0,
    y: 0,
    width,
    height,
  };
  const x = clampCropValue(
    area.x + Math.round(area.width * preset.x),
    0,
    Math.max(0, width - 1),
  );
  const y = clampCropValue(
    area.y + Math.round(area.height * preset.y),
    0,
    Math.max(0, height - 1),
  );
  const maxWidth = Math.max(1, area.x + area.width - x);
  const maxHeight = Math.max(1, area.y + area.height - y);

  return {
    x,
    y,
    width: clampCropValue(Math.round(area.width * preset.width), 1, maxWidth),
    height: clampCropValue(Math.round(area.height * preset.height), 1, maxHeight),
  };
}

function getAveragePatchColor(
  data: Uint8ClampedArray,
  width: number,
  startX: number,
  startY: number,
  patchWidth: number,
  patchHeight: number,
) {
  let redSum = 0;
  let greenSum = 0;
  let blueSum = 0;
  let count = 0;

  for (let y = startY; y < startY + patchHeight; y += 1) {
    for (let x = startX; x < startX + patchWidth; x += 1) {
      const pixelIndex = (y * width + x) * 4;
      redSum += data[pixelIndex];
      greenSum += data[pixelIndex + 1];
      blueSum += data[pixelIndex + 2];
      count += 1;
    }
  }

  if (count === 0) {
    return {
      red: 0,
      green: 0,
      blue: 0,
    };
  }

  return {
    red: redSum / count,
    green: greenSum / count,
    blue: blueSum / count,
  };
}

function estimateForegroundBounds(source: ImageBitmap) {
  const scale = Math.min(
    1,
    FOREGROUND_ANALYSIS_MAX_SIDE / Math.max(source.width, source.height),
  );
  const analysisWidth = Math.max(1, Math.round(source.width * scale));
  const analysisHeight = Math.max(1, Math.round(source.height * scale));
  const canvas = document.createElement('canvas');
  canvas.width = analysisWidth;
  canvas.height = analysisHeight;
  const context = canvas.getContext('2d', { alpha: false, willReadFrequently: true });

  if (!context) {
    releaseCanvas(canvas);
    return null;
  }

  context.drawImage(source, 0, 0, analysisWidth, analysisHeight);
  const { data } = context.getImageData(0, 0, analysisWidth, analysisHeight);
  const patchSize = Math.max(6, Math.round(Math.min(analysisWidth, analysisHeight) * 0.08));
  const patches = [
    getAveragePatchColor(data, analysisWidth, 0, 0, patchSize, patchSize),
    getAveragePatchColor(data, analysisWidth, analysisWidth - patchSize, 0, patchSize, patchSize),
    getAveragePatchColor(
      data,
      analysisWidth,
      0,
      analysisHeight - patchSize,
      patchSize,
      patchSize,
    ),
    getAveragePatchColor(
      data,
      analysisWidth,
      analysisWidth - patchSize,
      analysisHeight - patchSize,
      patchSize,
      patchSize,
    ),
  ];

  const background = patches.reduce(
    (current, patch) => ({
      red: current.red + patch.red,
      green: current.green + patch.green,
      blue: current.blue + patch.blue,
    }),
    {
      red: 0,
      green: 0,
      blue: 0,
    },
  );

  const backgroundRed = background.red / patches.length;
  const backgroundGreen = background.green / patches.length;
  const backgroundBlue = background.blue / patches.length;

  let left = analysisWidth;
  let top = analysisHeight;
  let right = -1;
  let bottom = -1;
  let foregroundCount = 0;

  for (let y = 0; y < analysisHeight; y += 1) {
    for (let x = 0; x < analysisWidth; x += 1) {
      const pixelIndex = (y * analysisWidth + x) * 4;
      const red = data[pixelIndex];
      const green = data[pixelIndex + 1];
      const blue = data[pixelIndex + 2];
      const colorDistance =
        Math.abs(red - backgroundRed) +
        Math.abs(green - backgroundGreen) +
        Math.abs(blue - backgroundBlue);
      const chroma = Math.max(red, green, blue) - Math.min(red, green, blue);

      if (
        colorDistance < FOREGROUND_COLOR_DISTANCE_THRESHOLD &&
        chroma < FOREGROUND_CHROMA_THRESHOLD
      ) {
        continue;
      }

      foregroundCount += 1;
      left = Math.min(left, x);
      top = Math.min(top, y);
      right = Math.max(right, x);
      bottom = Math.max(bottom, y);
    }
  }

  releaseCanvas(canvas);

  if (foregroundCount === 0 || right <= left || bottom <= top) {
    return null;
  }

  const widthCoverage = (right - left + 1) / analysisWidth;
  const heightCoverage = (bottom - top + 1) / analysisHeight;

  if (widthCoverage < 0.2 || heightCoverage < 0.15) {
    return null;
  }

  const paddingX = Math.max(4, Math.round((right - left + 1) * 0.08));
  const paddingY = Math.max(4, Math.round((bottom - top + 1) * 0.12));
  const paddedLeft = clampCropValue(left - paddingX, 0, Math.max(0, analysisWidth - 1));
  const paddedTop = clampCropValue(top - paddingY, 0, Math.max(0, analysisHeight - 1));
  const paddedRight = clampCropValue(right + paddingX, paddedLeft + 1, analysisWidth);
  const paddedBottom = clampCropValue(bottom + paddingY, paddedTop + 1, analysisHeight);

  return {
    x: clampCropValue(Math.floor(paddedLeft / scale), 0, Math.max(0, source.width - 1)),
    y: clampCropValue(Math.floor(paddedTop / scale), 0, Math.max(0, source.height - 1)),
    width: clampCropValue(
      Math.ceil((paddedRight - paddedLeft) / scale),
      1,
      source.width,
    ),
    height: clampCropValue(
      Math.ceil((paddedBottom - paddedTop) / scale),
      1,
      source.height,
    ),
  };
}

function getUploadCropCandidates(width: number, height: number, baseRect?: CropRect) {
  const activeRect = baseRect ?? {
    x: 0,
    y: 0,
    width,
    height,
  };
  const stripAspectRatio = activeRect.width / Math.max(1, activeRect.height);
  const hasForegroundCrop =
    activeRect.x > 0 ||
    activeRect.y > 0 ||
    activeRect.width < width ||
    activeRect.height < height;
  const foregroundCandidate = hasForegroundCrop
    ? ({
        id: 'foreground-bounds',
        label: 'texto detectado',
        rect: activeRect,
      } satisfies UploadCropCandidate)
    : null;
  const stripTopBandCandidate =
    stripAspectRatio >= 2.3
      ? ({
          id: 'strip-top-band',
          label: 'franja superior del serial',
          rect: {
            x: activeRect.x,
            y: activeRect.y,
            width: activeRect.width,
            height: clampCropValue(
              Math.round(activeRect.height * 0.68),
              1,
              Math.max(1, height - activeRect.y),
            ),
          },
        } satisfies UploadCropCandidate)
      : null;
  const fullImageCandidate = {
    id: 'full-image',
    label: 'imagen completa',
    rect: {
      x: 0,
      y: 0,
      width,
      height,
    },
  } satisfies UploadCropCandidate;
  const presetCandidates = UPLOAD_SCAN_PRESETS.map<UploadCropCandidate>((preset) => ({
    id: preset.id,
    label: preset.label,
    rect: resolveRelativeCropRect(width, height, preset, activeRect),
  }));

  const guideWidth = Math.max(1, Math.round(activeRect.width * CAMERA_FRAME_WIDTH_RATIO));
  const guideHeight = Math.max(1, Math.round(activeRect.height * CAMERA_FRAME_HEIGHT_RATIO));
  const guideRect = {
    x: clampCropValue(
      activeRect.x + Math.round((activeRect.width - guideWidth) / 2),
      0,
      Math.max(0, width - 1),
    ),
    y: clampCropValue(
      activeRect.y + Math.round((activeRect.height - guideHeight) / 2),
      0,
      Math.max(0, height - 1),
    ),
    width: clampCropValue(guideWidth, 1, Math.max(1, width - activeRect.x)),
    height: clampCropValue(guideHeight, 1, Math.max(1, height - activeRect.y)),
  };
  const topGuideHeight = Math.max(1, Math.round(activeRect.height * 0.22));
  const topGuideY = clampCropValue(
    activeRect.y + Math.round(activeRect.height * 0.04),
    0,
    Math.max(0, height - 1),
  );

  return [
    ...(foregroundCandidate ? [foregroundCandidate] : []),
    ...(stripTopBandCandidate ? [stripTopBandCandidate] : []),
    fullImageCandidate,
    ...presetCandidates,
    {
      id: 'top-guide',
      label: 'guía superior ancha',
      rect: {
            x: guideRect.x,
            y: topGuideY,
            width: guideRect.width,
            height: clampCropValue(
              topGuideHeight,
              1,
              Math.max(1, activeRect.y + activeRect.height - topGuideY),
            ),
          },
        },
    {
      id: 'center-guide',
      label: 'guía central',
      rect: guideRect,
    },
  ] satisfies UploadCropCandidate[];
}

function releaseCanvas(canvas: HTMLCanvasElement | null) {
  if (!canvas) {
    return;
  }

  const context = canvas.getContext('2d');
  if (context) {
    context.clearRect(0, 0, canvas.width, canvas.height);
  }

  canvas.width = 0;
  canvas.height = 0;
}

function canvasToBlob(canvas: HTMLCanvasElement) {
  return new Promise<Blob | null>((resolve) => {
    canvas.toBlob(resolve, 'image/jpeg', DERIVATIVE_JPEG_QUALITY);
  });
}

async function createCroppedBitmap(source: ImageBitmap, rect: CropRect) {
  const canvas = document.createElement('canvas');
  canvas.width = rect.width;
  canvas.height = rect.height;
  const context = canvas.getContext('2d', { alpha: false, willReadFrequently: true });

  if (!context) {
    releaseCanvas(canvas);
    throw new Error('No se pudo preparar el recorte para el OCR.');
  }

  context.drawImage(
    source,
    rect.x,
    rect.y,
    rect.width,
    rect.height,
    0,
    0,
    rect.width,
    rect.height,
  );

  try {
    const [previewBlob, imageBitmap] = await Promise.all([
      canvasToBlob(canvas),
      createImageBitmap(canvas),
    ]);

    return {
      imageBitmap,
      previewBlob,
    };
  } finally {
    releaseCanvas(canvas);
  }
}

function extractSerialNumber(serialDisplay: string) {
  const numericText = serialDisplay.replace(/[^0-9]/g, '');
  const serialNumeric = Number(numericText);

  return Number.isFinite(serialNumeric) ? serialNumeric : null;
}

function isReportedAsInvalid(serialNumeric: number) {
  return validRanges.some(([min, max]) => serialNumeric >= min && serialNumeric <= max);
}

function toSerialResults(serials: DetectedSerial[]) {
  return serials
    .map<SerialResult | null>((serial) => {
      const serialNumeric = extractSerialNumber(serial.value);
      if (serialNumeric === null) {
        return null;
      }

      return {
        isValid: !isReportedAsInvalid(serialNumeric),
        serialDisplay: serial.value,
        serialNumeric,
        source: serial.source,
        confidence: serial.confidence,
      };
    })
    .filter((result): result is SerialResult => result !== null);
}

function parseExpectedSerialFromFileName(fileName: string) {
  const match = fileName.match(/(?:^|[^0-9])(\d{8,9})(?=\D|$)/);
  return match?.[1] ?? null;
}

function getDiagnosticStatusLabel(result: DiagnosticResult) {
  if (result.error) {
    return result.error;
  }

  if (result.detected) {
    return result.detected;
  }

  if (result.feedback === 'low-confidence') {
    return 'Lectura incierta';
  }

  if (result.feedback === 'not-found') {
    return 'Patrón no encontrado';
  }

  return 'Sin coincidencia';
}

function workerErrorMessage(error: unknown, fallback: string) {
  if (error instanceof Error && error.message) {
    return error.message;
  }

  return fallback;
}

export default function App() {
  const [ocrStatus, setOcrStatus] = useState<OcrStatus>('idle');
  const [ocrInitError, setOcrInitError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDiagnosticsVisible, setIsDiagnosticsVisible] = useState(false);
  const [isDiagnosticsRunning, setIsDiagnosticsRunning] = useState(false);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [results, setResults] = useState<SerialResult[] | null>(null);
  const [scanFeedback, setScanFeedback] = useState<ScanFeedback>('none');
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isStartingCamera, setIsStartingCamera] = useState(false);
  const [isTorchAvailable, setIsTorchAvailable] = useState(false);
  const [isTorchEnabled, setIsTorchEnabled] = useState(false);
  const [isTogglingTorch, setIsTogglingTorch] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [warmupProgress, setWarmupProgress] = useState<LoadingProgressState | null>(null);
  const [scanProgress, setScanProgress] = useState<LoadingProgressState | null>(null);
  const [diagnosticsProgress, setDiagnosticsProgress] = useState<LoadingProgressState | null>(
    null,
  );
  const [diagnosticResults, setDiagnosticResults] = useState<DiagnosticResult[]>([]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const diagnosticInputRef = useRef<HTMLInputElement>(null);
  const warmupPromiseRef = useRef<Promise<void> | null>(null);
  const activeJobRef = useRef(0);
  const previewUrlRef = useRef<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const footerTapCountRef = useRef(0);
  const footerTapResetRef = useRef<number | null>(null);
  const isBusy = isProcessing || isDiagnosticsRunning;

  function revokePreviewUrl() {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
  }

  function setPreviewFromBlob(blob: Blob) {
    revokePreviewUrl();

    const previewUrl = URL.createObjectURL(blob);
    previewUrlRef.current = previewUrl;
    setImageSrc(previewUrl);
  }

  function clearScanOutput() {
    setResults(null);
    setScanFeedback('none');
  }

  function clearFooterTapResetTimer() {
    if (footerTapResetRef.current !== null) {
      window.clearTimeout(footerTapResetRef.current);
      footerTapResetRef.current = null;
    }
  }

  function handleFooterNoteTap() {
    clearFooterTapResetTimer();
    footerTapCountRef.current += 1;

    if (footerTapCountRef.current >= 5) {
      footerTapCountRef.current = 0;
      setIsDiagnosticsVisible((current) => !current);
      return;
    }

    footerTapResetRef.current = window.setTimeout(() => {
      footerTapCountRef.current = 0;
      footerTapResetRef.current = null;
    }, 1500);
  }

  function ensureActiveJob(jobId: number) {
    if (activeJobRef.current !== jobId) {
      throw new Error('Escaneo cancelado.');
    }
  }

  function getActiveVideoTrack() {
    const stream = streamRef.current;
    if (!stream) {
      return null;
    }

    return (stream.getVideoTracks()[0] as TorchCapableTrack | undefined) ?? null;
  }

  function supportsTorchControl(track: TorchCapableTrack | null) {
    if (!track || typeof track.getCapabilities !== 'function') {
      return false;
    }

    const supportedConstraints = navigator.mediaDevices?.getSupportedConstraints?.() as
      | (MediaTrackSupportedConstraints & TorchSettings)
      | undefined;

    if (!supportedConstraints?.torch) {
      return false;
    }

    const capabilities = track.getCapabilities() as MediaTrackCapabilities & TorchSettings;
    return Boolean(capabilities?.torch);
  }

  function syncTorchAvailability() {
    const track = getActiveVideoTrack();
    const supportsTorch = supportsTorchControl(track);

    setIsTorchAvailable(supportsTorch);
    if (!supportsTorch) {
      setIsTorchEnabled(false);
      setIsTogglingTorch(false);
    }
  }

  function stopCamera() {
    const stream = streamRef.current;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
    }

    setIsCameraActive(false);
    setIsStartingCamera(false);
    setIsTorchAvailable(false);
    setIsTorchEnabled(false);
    setIsTogglingTorch(false);
  }

  async function toggleTorch() {
    const track = getActiveVideoTrack();
    const supportsTorch = supportsTorchControl(track);

    if (!track || !supportsTorch || isTogglingTorch) {
      return;
    }

    setIsTogglingTorch(true);
    const nextTorchState = !isTorchEnabled;

    try {
      await track.applyConstraints({
        advanced: [{ torch: nextTorchState } as MediaTrackConstraintSet & TorchSettings],
      } as MediaTrackConstraints);

      setIsTorchEnabled(nextTorchState);
      setCameraError(null);
    } catch (error) {
      console.error('Error al cambiar el flash:', error);
      setCameraError('No se pudo cambiar el flash de la cámara.');
    } finally {
      setIsTogglingTorch(false);
    }
  }

  function warmupOcr() {
    if (ocrStatus === 'ready') {
      return Promise.resolve();
    }

    if (warmupPromiseRef.current) {
      return warmupPromiseRef.current;
    }

    setOcrStatus('initializing');
    setOcrInitError(null);
    setWarmupProgress({
      progress: null,
      label: 'Cargando motor OCR',
    });

    const promise = ensurePaddleOcrReady()
      .then(() => {
        setOcrStatus('ready');
        setOcrInitError(null);
      })
      .catch((error) => {
        const message = workerErrorMessage(
          error,
          'No se pudo inicializar el motor OCR en este dispositivo.',
        );
        setOcrStatus('error');
        setOcrInitError(message);
        throw error;
      })
      .finally(() => {
        setWarmupProgress(null);
        warmupPromiseRef.current = null;
      });

    warmupPromiseRef.current = promise;
    return promise;
  }

  async function runOcrScan(
    image: ImageBitmap,
    progressLabel: string,
    detail?: string,
    updateProgress: (progress: LoadingProgressState | null) => void = setScanProgress,
  ) {
    await warmupOcr();
    updateProgress({
      progress: null,
      label: progressLabel,
      detail,
    });
    await waitForNextPaint(1);
    return scanWithPaddleOcr(image);
  }

  function consolidateDetectedSerials(
    acceptedSerials: Map<string, DetectedSerial>,
    candidateVotes: Map<string, CandidateVote>,
    strongestFeedback: ScanFeedback,
  ): AggregatedScanResult {
    if (acceptedSerials.size > 0) {
      return {
        serials: [...acceptedSerials.values()].sort(
          (left, right) => right.confidence - left.confidence,
        ),
        feedback: 'none',
      };
    }

    const consensusSerials = [...candidateVotes.entries()]
      .map(([value, vote]) => ({
        value,
        confidence: vote.confidenceTotal / vote.hits,
        hits: vote.hits,
        source: vote.source,
      }))
      .filter((vote) => vote.hits >= 2 && vote.confidence >= 0.5)
      .sort((left, right) => right.confidence - left.confidence)
      .map<DetectedSerial>((vote) => ({
        value: vote.value,
        confidence: vote.confidence,
        source: vote.source,
      }));

    if (consensusSerials.length > 0) {
      return {
        serials: consensusSerials,
        feedback: 'none',
      };
    }

    return {
      serials: [],
      feedback: strongestFeedback === 'none' ? 'not-found' : strongestFeedback,
    };
  }

  async function scanUploadCandidates(
    sourceBitmap: ImageBitmap,
    jobId: number,
    updateProgress: (progress: LoadingProgressState | null) => void,
    options?: {
      previewFallback?: Blob;
      onPreview?: (blob: Blob) => void;
    },
  ) {
    const foregroundBounds = estimateForegroundBounds(sourceBitmap);
    const uploadCandidates = getUploadCropCandidates(
      sourceBitmap.width,
      sourceBitmap.height,
      foregroundBounds ?? undefined,
    );
    const candidates =
      sourceBitmap.height <= 96 || sourceBitmap.width <= 320
        ? uploadCandidates.filter((candidate) =>
            ['foreground-bounds', 'strip-top-band', 'full-image'].includes(candidate.id),
          )
        : uploadCandidates;
    let strongestFeedback: ScanFeedback = 'none';
    const acceptedSerials = new Map<string, DetectedSerial>();
    const candidateVotes = new Map<string, CandidateVote>();

    for (let index = 0; index < candidates.length; index += 1) {
      ensureActiveJob(jobId);

      const candidate = candidates[index];
      updateProgress({
        progress: null,
        label: `Probando recorte ${index + 1} de ${candidates.length}`,
        detail: candidate.label,
      });

      let imageBitmap: ImageBitmap | null = null;

      try {
        const cropped = await createCroppedBitmap(sourceBitmap, candidate.rect);
        imageBitmap = cropped.imageBitmap;

        const previewBlob = cropped.previewBlob ?? options?.previewFallback;
        if (previewBlob && options?.onPreview) {
          options.onPreview(previewBlob);
        }

        const scan = await runOcrScan(
          imageBitmap,
          `Analizando recorte ${index + 1} de ${candidates.length}`,
          candidate.label,
          updateProgress,
        );

        ensureActiveJob(jobId);

        for (const serial of scan.serials) {
          const current = acceptedSerials.get(serial.value);
          if (!current || serial.confidence > current.confidence) {
            acceptedSerials.set(serial.value, serial);
          }
        }

        for (const candidateSerial of scan.candidateSerials) {
          const current = candidateVotes.get(candidateSerial.value);

          if (!current) {
            candidateVotes.set(candidateSerial.value, {
              confidenceTotal: candidateSerial.confidence,
              hits: 1,
              maxConfidence: candidateSerial.confidence,
              source: candidateSerial.source,
            });
            continue;
          }

          current.confidenceTotal += candidateSerial.confidence;
          current.hits += 1;
          current.maxConfidence = Math.max(current.maxConfidence, candidateSerial.confidence);

          if (candidateSerial.confidence >= current.maxConfidence) {
            current.source = candidateSerial.source;
          }
        }

        if (scan.feedback === 'low-confidence') {
          strongestFeedback = 'low-confidence';
        } else if (strongestFeedback === 'none') {
          strongestFeedback = scan.feedback;
        }
      } finally {
        imageBitmap?.close();
      }
    }

    return consolidateDetectedSerials(acceptedSerials, candidateVotes, strongestFeedback);
  }

  async function processCapturedBitmap(imageBitmap: ImageBitmap) {
    const jobId = ++activeJobRef.current;

    setIsProcessing(true);
    setScanProgress({
      progress: null,
      label: 'Preparando escaneo',
    });

    try {
      await waitForNextPaint(2);
      if (activeJobRef.current !== jobId) {
        return;
      }

      const scan = await runOcrScan(imageBitmap, 'Analizando serial');

      if (activeJobRef.current !== jobId) {
        return;
      }

      setResults(toSerialResults(scan.serials));
      setScanFeedback(scan.feedback);
      setCameraError(null);
    } catch (error) {
      if (activeJobRef.current !== jobId) {
        return;
      }

      if (error instanceof Error && error.message === 'Escaneo cancelado.') {
        return;
      }

      console.error('Error de OCR:', error);
      setResults(null);
      setScanFeedback('none');

      if (error instanceof PaddleOcrTimeoutError) {
        setCameraError(
          'El OCR tardó demasiado en responder. Cierra otras apps y vuelve a intentarlo.',
        );
      } else if (looksLikePaddleMemoryError(error)) {
        setCameraError(
          'El dispositivo se quedó sin memoria durante el escaneo. Intenta de nuevo con el serial más cerca.',
        );
      } else {
        setCameraError(
          workerErrorMessage(error, 'No se pudo completar el escaneo. Intenta de nuevo.'),
        );
      }
    } finally {
      try {
        imageBitmap.close();
      } catch {
        // Ignored.
      }

      if (activeJobRef.current === jobId) {
        setScanProgress(null);
        setIsProcessing(false);
      }
    }
  }

  async function processUploadedImage(file: File) {
    if (isBusy) {
      return;
    }

    if (typeof createImageBitmap !== 'function') {
      setCameraError('Este navegador no soporta el procesamiento OCR necesario.');
      return;
    }

    stopCamera();
    clearScanOutput();
    setCameraError(null);

    const jobId = ++activeJobRef.current;
    let sourceBitmap: ImageBitmap | null = null;

    setIsProcessing(true);
    setScanProgress({
      progress: null,
      label: 'Preparando imagen de prueba',
      detail: file.name,
    });

    try {
      sourceBitmap = await createImageBitmap(file);
      ensureActiveJob(jobId);

      const aggregated = await scanUploadCandidates(sourceBitmap, jobId, setScanProgress, {
        previewFallback: file,
        onPreview: setPreviewFromBlob,
      });

      ensureActiveJob(jobId);

      const serialResults = toSerialResults(aggregated.serials);
      setResults(serialResults);
      setScanFeedback(aggregated.feedback);
    } catch (error) {
      if (error instanceof Error && error.message === 'Escaneo cancelado.') {
        return;
      }

      if (activeJobRef.current !== jobId) {
        return;
      }

      console.error('Error al procesar la imagen:', error);
      setResults(null);
      setScanFeedback('none');
      if (error instanceof PaddleOcrTimeoutError) {
        setCameraError(
          'El OCR tardó demasiado en responder. Intenta de nuevo con una imagen más nítida y recortada.',
        );
      } else if (looksLikePaddleMemoryError(error)) {
        setCameraError(
          'El dispositivo se quedó sin memoria durante el análisis. Usa una imagen más ajustada al serial.',
        );
      } else {
        setCameraError(
          workerErrorMessage(error, 'No se pudo procesar la imagen seleccionada. Intenta de nuevo.'),
        );
      }
    } finally {
      sourceBitmap?.close();

      if (activeJobRef.current === jobId) {
        setScanProgress(null);
        setIsProcessing(false);
      }
    }
  }

  function handleImageFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const [file] = event.target.files ?? [];
    event.currentTarget.value = '';

    if (!file) {
      return;
    }

    void processUploadedImage(file);
  }

  async function runDiagnostics(files: File[]) {
    if (files.length === 0 || isBusy) {
      return;
    }

    if (typeof createImageBitmap !== 'function') {
      setCameraError('Este navegador no soporta el procesamiento OCR necesario.');
      return;
    }

    stopCamera();
    setCameraError(null);
    setDiagnosticResults([]);

    const jobId = ++activeJobRef.current;
    const nextResults: DiagnosticResult[] = [];

    setIsDiagnosticsRunning(true);
    setDiagnosticsProgress({
      progress: null,
      label: 'Preparando diagnóstico',
      detail: `${files.length} archivo${files.length === 1 ? '' : 's'}`,
    });

    try {
      for (let index = 0; index < files.length; index += 1) {
        ensureActiveJob(jobId);

        const file = files[index];
        const expected = parseExpectedSerialFromFileName(file.name);
        let sourceBitmap: ImageBitmap | null = null;

        try {
          setDiagnosticsProgress({
            progress: null,
            label: `Analizando prueba ${index + 1} de ${files.length}`,
            detail: file.name,
          });

          sourceBitmap = await createImageBitmap(file);
          ensureActiveJob(jobId);

          const aggregated = await scanUploadCandidates(
            sourceBitmap,
            jobId,
            setDiagnosticsProgress,
          );
          ensureActiveJob(jobId);

          const serialResults = toSerialResults(aggregated.serials);
          const bestResult = serialResults[0] ?? null;
          const detected = bestResult?.serialDisplay ?? null;
          nextResults.push({
            fileName: file.name,
            expected,
            detected,
            confidence: bestResult?.confidence ?? null,
            feedback: aggregated.feedback,
            matched: expected ? detected === expected : null,
            error: null,
          });
        } catch (error) {
          if (error instanceof Error && error.message === 'Escaneo cancelado.') {
            throw error;
          }

          nextResults.push({
            fileName: file.name,
            expected,
            detected: null,
            confidence: null,
            feedback: 'not-found',
            matched: expected ? false : null,
            error: workerErrorMessage(error, 'No se pudo analizar esta imagen.'),
          });
        } finally {
          sourceBitmap?.close();
          setDiagnosticResults([...nextResults]);
        }
      }
    } finally {
      if (activeJobRef.current === jobId) {
        setDiagnosticsProgress(null);
        setIsDiagnosticsRunning(false);
      }
    }
  }

  function handleDiagnosticFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    event.currentTarget.value = '';

    if (files.length === 0) {
      return;
    }

    void runDiagnostics(files);
  }

  function beginScanFromCapture(imageBitmap: ImageBitmap, previewBlob: Blob | null) {
    clearScanOutput();

    if (previewBlob) {
      setPreviewFromBlob(previewBlob);
    } else {
      revokePreviewUrl();
      setImageSrc(null);
    }

    setCameraError(null);
    void processCapturedBitmap(imageBitmap);
  }

  async function startCamera() {
    if (isBusy || isStartingCamera) {
      return;
    }

    if (streamRef.current) {
      setCameraError(null);
      setIsCameraActive(true);
      syncTorchAvailability();
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Tu navegador no permite abrir la cámara.');
      return;
    }

    setIsStartingCamera(true);
    setCameraError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      streamRef.current = stream;
      setIsCameraActive(true);
      setIsTorchEnabled(false);
      syncTorchAvailability();
    } catch (error) {
      console.error('Error al abrir la cámara:', error);
      setCameraError('No se pudo abrir la cámara. Revisa los permisos e inténtalo de nuevo.');
    } finally {
      setIsStartingCamera(false);
    }
  }

  async function captureCameraFrame() {
    if (isBusy) {
      return;
    }

    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) {
      setCameraError('La cámara todavía se está preparando. Intenta de nuevo en un segundo.');
      return;
    }

    if (typeof createImageBitmap !== 'function') {
      setCameraError('Este navegador no soporta el procesamiento OCR necesario.');
      return;
    }

    const guideRect = getCaptureGuideRect(video.videoWidth, video.videoHeight);
    const canvas = document.createElement('canvas');
    canvas.width = guideRect.width;
    canvas.height = guideRect.height;
    const context = canvas.getContext('2d', { alpha: false, willReadFrequently: true });

    if (!context) {
      releaseCanvas(canvas);
      setCameraError('No se pudo preparar la captura. Intenta de nuevo.');
      return;
    }

    context.drawImage(
      video,
      guideRect.x,
      guideRect.y,
      guideRect.width,
      guideRect.height,
      0,
      0,
      guideRect.width,
      guideRect.height,
    );

    try {
      const [previewBlob, imageBitmap] = await Promise.all([
        canvasToBlob(canvas),
        createImageBitmap(canvas),
      ]);

      stopCamera();
      beginScanFromCapture(imageBitmap, previewBlob);
    } catch (error) {
      console.error('Error al capturar la imagen:', error);
      setCameraError('No se pudo procesar la foto del serial. Intenta de nuevo.');
    } finally {
      releaseCanvas(canvas);
    }
  }

  useEffect(() => {
    let isMounted = true;

    setOcrStatus('initializing');
    setOcrInitError(null);
    setWarmupProgress({
      progress: null,
      label: 'Preparando OCR',
    });

    const initialWarmup = ensurePaddleOcrReady()
      .then(() => {
        if (!isMounted) {
          return;
        }

        setOcrStatus('ready');
        setOcrInitError(null);
      })
      .catch((error) => {
        if (!isMounted) {
          return;
        }

        const message = workerErrorMessage(
          error,
          'No se pudo inicializar el motor OCR en este dispositivo.',
        );
        setOcrStatus('error');
        setOcrInitError(message);
      })
      .finally(() => {
        if (warmupPromiseRef.current === initialWarmup) {
          warmupPromiseRef.current = null;
        }

        if (isMounted) {
          setWarmupProgress(null);
        }
      });

    warmupPromiseRef.current = initialWarmup;

    return () => {
      isMounted = false;
      activeJobRef.current += 1;
      clearFooterTapResetTimer();
      revokePreviewUrl();
      setWarmupProgress(null);
      setScanProgress(null);
      setDiagnosticsProgress(null);
      const stream = streamRef.current;
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    const stream = streamRef.current;

    if (!isCameraActive || !video || !stream) {
      return;
    }

    if (video.srcObject !== stream) {
      video.srcObject = stream;
    }

    if (isProcessing) {
      video.pause();
      return;
    }

    void video.play().catch(noop);
  }, [isCameraActive, isProcessing]);

  useEffect(() => {
    if (isProcessing || (results === null && scanFeedback === 'none')) {
      return;
    }

    void waitForNextPaint(2).then(() => {
      window.scrollTo({
        top: document.documentElement.scrollHeight,
        behavior: 'smooth',
      });
    });
  }, [isProcessing, results, scanFeedback]);

  const diagnosticComparableCount = diagnosticResults.filter(
    (result) => result.expected !== null,
  ).length;
  const diagnosticPassCount = diagnosticResults.filter((result) => result.matched === true).length;

  return (
    <div className="container">
      <main className="content">
        <input
          ref={diagnosticInputRef}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={handleDiagnosticFileChange}
        />
        {ocrStatus === 'initializing' && !isCameraActive ? (
          <div className="status-box initializing">
            <Loader2 className="spinner" size={32} />
            <p>
              Cargando lector OCR
              <br />
              <small>La primera carga puede tardar unos segundos</small>
            </p>
            <LoadingProgressBar progress={warmupProgress} />
          </div>
        ) : ocrStatus === 'error' && !isCameraActive ? (
          <div className="status-box initializing">
            <XCircle size={32} className="icon-invalid" />
            <p>
              No se pudo cargar el lector OCR
              <br />
              <small>{ocrInitError ?? 'Intenta nuevamente.'}</small>
            </p>
            <button className="primary-btn" onClick={() => void warmupOcr()}>
              Reintentar
            </button>
          </div>
        ) : (
          <section className="camera-card">
            <div className="camera-copy">
              <h2>Coloca el número de serie dentro del recuadro</h2>
            </div>

            {isCameraActive && (
              <div className="camera-stage-shell">
                <div className="camera-stage">
                  <video ref={videoRef} className="camera-video" autoPlay muted playsInline />
                  <div className="camera-guide" />
                  <div className="camera-guide-label">Alinea el número de serie aquí</div>
                </div>
              </div>
            )}

            {cameraError && <p className="camera-error">{cameraError}</p>}

            <div className="camera-actions">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                hidden
                onChange={handleImageFileChange}
              />

              <button
                type="button"
                className="primary-btn"
                disabled={isBusy || isStartingCamera || ocrStatus === 'initializing'}
                onClick={() => {
                  if (isCameraActive) {
                    void captureCameraFrame();
                  } else {
                    void startCamera();
                  }
                }}
              >
                <Camera size={20} />
                <span>
                  {isCameraActive
                    ? 'Tomar foto del serial'
                    : isStartingCamera
                      ? 'Abriendo cámara...'
                      : 'Abrir cámara guiada'}
                </span>
              </button>

              <button
                type="button"
                className="secondary-btn"
                disabled={isBusy || isStartingCamera || ocrStatus === 'initializing'}
                onClick={() => fileInputRef.current?.click()}
              >
                <ImageUp size={20} />
                <span>Probar con imagen</span>
              </button>

              {isCameraActive && isTorchAvailable && (
                <button
                  type="button"
                  className="secondary-btn"
                  disabled={isBusy || isTogglingTorch}
                  onClick={() => void toggleTorch()}
                >
                  {isTogglingTorch
                    ? 'Cambiando flash...'
                    : isTorchEnabled
                      ? 'Apagar flash'
                      : 'Encender flash'}
                </button>
              )}

              {isCameraActive && (
                <button
                  type="button"
                  className="secondary-btn"
                  disabled={isBusy}
                  onClick={stopCamera}
                >
                  Cerrar cámara
                </button>
              )}
            </div>
          </section>
        )}

        {imageSrc && (
          <div className="preview-card">
            <img key={imageSrc} src={imageSrc} alt="Billete escaneado" className="scanned-image" />

            {isProcessing && !isCameraActive && (
              <div className="processing-overlay">
                <Loader2 className="spinner" size={48} />
                <span>Escaneando serial...</span>
                <LoadingProgressBar progress={scanProgress} />
              </div>
            )}
          </div>
        )}

        {!isProcessing && results !== null && results.length > 0 && (
          <div className="results-container">
            {results.map((result) => (
              <div
                key={`${result.serialDisplay}-${result.source}-${result.confidence}`}
                className={`result-card ${result.isValid ? 'valid' : 'invalid'}`}
              >
                {result.isValid ? (
                  <>
                    <CheckCircle2 size={48} className="icon-valid" />
                    <h2>Billete Válido</h2>
                    <p className="serial-code">{result.serialDisplay}</p>
                    <p>
                      El número {result.serialDisplay} no pertenece a los rangos
                      reportados por el BCB.
                    </p>
                  </>
                ) : (
                  <>
                    <XCircle size={48} className="icon-invalid" />
                    <h2>Billete Inválido</h2>
                    <p className="serial-code">{result.serialDisplay}</p>
                    <p>
                      ¡Cuidado! El número {result.serialDisplay} pertenece a un lote reportado robado
                      por el BCB.
                    </p>
                  </>
                )}
              </div>
            ))}
          </div>
        )}

        {!isProcessing && results !== null && results.length === 0 && scanFeedback === 'low-confidence' && (
          <div className="result-card invalid">
            <XCircle size={48} className="icon-invalid" />
            <h2>Lectura incierta</h2>
            <p>No se pudo leer con suficiente confianza el número de serie.</p>
            <p>Vuelve a intentar con el serial más cerca, mejor enfocado y dentro del recuadro.</p>
          </div>
        )}

        {!isProcessing && results !== null && results.length === 0 && scanFeedback === 'not-found' && (
          <div className="result-card invalid">
            <XCircle size={48} className="icon-invalid" />
            <h2>Patrón no encontrado</h2>
            <p>No se detectó un número de serie de 8 a 9 dígitos en la imagen capturada.</p>
          </div>
        )}

        {isDiagnosticsVisible && (
          <section className="diagnostics-card">
            <div className="diagnostics-header">
              <div className="diagnostics-copy">
                <span className="camera-badge">Oculto</span>
                <h2>Modo diagnóstico</h2>
                <p>
                  Ejecuta el mismo flujo real de recortes y OCR sobre varias imágenes. Si el
                  archivo incluye un serial de 8 a 9 dígitos en el nombre, la app lo compara
                  automáticamente.
                </p>
              </div>
              <button
                type="button"
                className="secondary-btn"
                disabled={isDiagnosticsRunning}
                onClick={() => setIsDiagnosticsVisible(false)}
              >
                Ocultar
              </button>
            </div>

            <div className="diagnostics-actions">
              <button
                type="button"
                className="primary-btn"
                disabled={isBusy || ocrStatus === 'initializing'}
                onClick={() => diagnosticInputRef.current?.click()}
              >
                <ImageUp size={20} />
                <span>Cargar fixtures</span>
              </button>

              {diagnosticResults.length > 0 && (
                <button
                  type="button"
                  className="secondary-btn"
                  disabled={isDiagnosticsRunning}
                  onClick={() => setDiagnosticResults([])}
                >
                  Limpiar resultados
                </button>
              )}
            </div>

            {isDiagnosticsRunning && (
              <div className="diagnostics-progress">
                <Loader2 className="spinner" size={20} />
                <LoadingProgressBar progress={diagnosticsProgress} />
              </div>
            )}

            {diagnosticResults.length > 0 && (
              <>
                <div className="diagnostics-summary">
                  <strong>{diagnosticResults.length}</strong>
                  <span>archivos analizados</span>
                  {diagnosticComparableCount > 0 && (
                    <span>
                      {diagnosticPassCount}/{diagnosticComparableCount} coincidencias exactas
                    </span>
                  )}
                </div>

                <div className="diagnostics-list">
                  {diagnosticResults.map((result) => (
                    <article
                      key={`${result.fileName}-${result.expected ?? 'na'}`}
                      className={`diagnostic-row ${
                        result.matched === true
                          ? 'diagnostic-pass'
                          : result.matched === false
                            ? 'diagnostic-fail'
                            : 'diagnostic-neutral'
                      }`}
                    >
                      <div className="diagnostic-row-main">
                        <strong>{result.fileName}</strong>
                        <span>Esperado: {result.expected ?? 'sin referencia'}</span>
                      </div>
                      <div className="diagnostic-row-meta">
                        <span>{getDiagnosticStatusLabel(result)}</span>
                        {result.confidence !== null && (
                          <span>{Math.round(result.confidence * 100)}%</span>
                        )}
                      </div>
                    </article>
                  ))}
                </div>
              </>
            )}
          </section>
        )}
      </main>

      <footer className="footer-copyright">
        <button type="button" className="footer-note footer-note-trigger" onClick={handleFooterNoteTap}>
          Esta app corre en tu propio dispositivo, así que puede
          tardar más o confundirse cuando la foto está movida o está oscura.
        </button>
        <a href="https://github.com/nubol23" target="_blank" rel="noopener noreferrer">
          © {new Date().getFullYear()} nubol23
        </a>
      </footer>
    </div>
  );
}
