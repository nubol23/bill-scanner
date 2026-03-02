import { useEffect, useRef, useState } from 'react';
import { Camera, CheckCircle2, ImageUp, Loader2, XCircle } from 'lucide-react';

import { ScanApiError, scanSerialImage, type RecognizeResponse } from './lib/api';

const CAMERA_FRAME_WIDTH_RATIO = 0.78;
const CAMERA_FRAME_HEIGHT_RATIO = 0.18;
const DERIVATIVE_JPEG_QUALITY = 0.92;
const FOREGROUND_ANALYSIS_MAX_SIDE = 256;
const FOREGROUND_COLOR_DISTANCE_THRESHOLD = 44;
const FOREGROUND_CHROMA_THRESHOLD = 20;
const ACCEPTED_CONFIDENCE_THRESHOLD = 0.75;

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

type BillDenomination = '10' | '20' | '50';
type ScanFeedback = 'none' | 'not-found';

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
  confidence: number;
};

type LoadingProgressState = {
  progress: number | null;
  label: string;
  detail?: string;
};

const UPLOAD_SCAN_PRESETS: readonly RelativeCropPreset[] = [
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

const BILL_DENOMINATIONS = ['10', '20', '50'] as const satisfies readonly BillDenomination[];

const INVALID_RANGES_BY_DENOMINATION: Record<
  BillDenomination,
  ReadonlyArray<readonly [number, number]>
> = {
  '10': [
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
  ],
  '20': [
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
  ],
  '50': [
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
  ],
};

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
  const x = clampCropValue(area.x + Math.round(area.width * preset.x), 0, Math.max(0, width - 1));
  const y = clampCropValue(area.y + Math.round(area.height * preset.y), 0, Math.max(0, height - 1));
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
    return { red: 0, green: 0, blue: 0 };
  }

  return {
    red: redSum / count,
    green: greenSum / count,
    blue: blueSum / count,
  };
}

function estimateForegroundBounds(source: ImageBitmap) {
  const scale = Math.min(1, FOREGROUND_ANALYSIS_MAX_SIDE / Math.max(source.width, source.height));
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
    { red: 0, green: 0, blue: 0 },
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
    width: clampCropValue(Math.ceil((paddedRight - paddedLeft) / scale), 1, source.width),
    height: clampCropValue(Math.ceil((paddedBottom - paddedTop) / scale), 1, source.height),
  };
}

function getUploadCropCandidates(width: number, height: number, baseRect?: CropRect) {
  const activeRect = baseRect ?? { x: 0, y: 0, width, height };
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

  return [
    ...(foregroundCandidate ? [foregroundCandidate] : []),
    ...(stripTopBandCandidate ? [stripTopBandCandidate] : []),
    ...UPLOAD_SCAN_PRESETS.map<UploadCropCandidate>((preset) => ({
      id: preset.id,
      label: preset.label,
      rect: resolveRelativeCropRect(width, height, preset, activeRect),
    })),
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

async function createCroppedBlob(source: ImageBitmap, rect: CropRect) {
  const canvas = document.createElement('canvas');
  canvas.width = rect.width;
  canvas.height = rect.height;
  const context = canvas.getContext('2d', { alpha: false, willReadFrequently: true });

  if (!context) {
    releaseCanvas(canvas);
    throw new Error('No se pudo preparar el recorte.');
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
    const blob = await canvasToBlob(canvas);
    if (!blob) {
      throw new Error('No se pudo generar la imagen recortada.');
    }
    return blob;
  } finally {
    releaseCanvas(canvas);
  }
}

function extractSerialNumber(serialDisplay: string) {
  const numericText = serialDisplay.replace(/[^0-9]/g, '');
  const serialNumeric = Number(numericText);
  return Number.isFinite(serialNumeric) ? serialNumeric : null;
}

function isReportedAsInvalid(serialNumeric: number, denomination: BillDenomination) {
  return INVALID_RANGES_BY_DENOMINATION[denomination].some(
    ([min, max]) => serialNumeric >= min && serialNumeric <= max,
  );
}

function toSerialResult(
  serialDisplay: string,
  confidence: number,
  denomination: BillDenomination,
): SerialResult | null {
  const serialNumeric = extractSerialNumber(serialDisplay);
  if (serialNumeric === null) {
    return null;
  }

  return {
    isValid: !isReportedAsInvalid(serialNumeric, denomination),
    serialDisplay,
    serialNumeric,
    confidence,
  };
}

function bestResponse(
  current: RecognizeResponse | null,
  next: RecognizeResponse,
): RecognizeResponse {
  if (!current) {
    return next;
  }

  const currentConfidence = current.confidence ?? 0;
  const nextConfidence = next.confidence ?? 0;
  if (nextConfidence > currentConfidence) {
    return next;
  }

  return current;
}

function errorMessageForScan(error: unknown) {
  if (error instanceof ScanApiError) {
    switch (error.kind) {
      case 'timeout':
        return error.message;
      case 'unavailable':
        return error.message;
      case 'rate-limited':
        return 'Se alcanzó el límite temporal del servidor OCR. Espera unos segundos.';
      case 'invalid-image':
        return error.message;
      default:
        return 'El servidor OCR falló. Intenta de nuevo en unos segundos.';
    }
  }

  if (error instanceof Error && error.message) {
    return error.message;
  }

  return 'No se pudo completar el análisis remoto.';
}

export default function App() {
  const [selectedDenomination, setSelectedDenomination] = useState<BillDenomination>('50');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isStartingCamera, setIsStartingCamera] = useState(false);
  const [isTorchAvailable, setIsTorchAvailable] = useState(false);
  const [isTorchEnabled, setIsTorchEnabled] = useState(false);
  const [isTogglingTorch, setIsTogglingTorch] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [results, setResults] = useState<SerialResult[] | null>(null);
  const [scanFeedback, setScanFeedback] = useState<ScanFeedback>('none');
  const [scanProgress, setScanProgress] = useState<LoadingProgressState | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const previewUrlRef = useRef<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const isBusy = isProcessing || isStartingCamera || isTogglingTorch;

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
    } catch {
      setCameraError('No se pudo cambiar el flash de la cámara.');
    } finally {
      setIsTogglingTorch(false);
    }
  }

  async function applyRecognizeResponse(response: RecognizeResponse) {
    if (response.status === 'ok' && response.serial) {
      const nextResult = toSerialResult(
        response.serial,
        response.confidence ?? response.candidates[0]?.confidence ?? 0.5,
        selectedDenomination,
      );
      setResults(nextResult ? [nextResult] : []);
      setScanFeedback('none');
      setCameraError(null);
      return;
    }

    setResults([]);
    setScanFeedback('not-found');
    setCameraError(null);
  }

  async function runRemoteScan(blob: Blob, label: string, detail?: string) {
    setScanProgress({
      progress: null,
      label,
      detail,
    });
    await waitForNextPaint(1);
    return scanSerialImage(blob);
  }

  async function processDirectBlob(blob: Blob, label: string, detail?: string) {
    setIsProcessing(true);
    try {
      const response = await runRemoteScan(blob, label, detail);
      await applyRecognizeResponse(response);
    } catch (error) {
      setResults(null);
      setScanFeedback('none');
      setCameraError(errorMessageForScan(error));
    } finally {
      setScanProgress(null);
      setIsProcessing(false);
    }
  }

  async function startCamera() {
    if (isBusy) {
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
    } catch {
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
      const previewBlob = await canvasToBlob(canvas);
      if (!previewBlob) {
        throw new Error('No se pudo generar la captura.');
      }

      stopCamera();
      clearScanOutput();
      setPreviewFromBlob(previewBlob);
      setCameraError(null);
      await processDirectBlob(previewBlob, 'Consultando OCR remoto...', 'captura guiada');
    } catch (error) {
      setCameraError(errorMessageForScan(error));
    } finally {
      releaseCanvas(canvas);
    }
  }

  async function processUploadedImage(file: File) {
    if (isBusy) {
      return;
    }

    if (typeof createImageBitmap !== 'function') {
      setCameraError('Este navegador no soporta el procesamiento necesario.');
      return;
    }

    stopCamera();
    clearScanOutput();
    setCameraError(null);
    setIsProcessing(true);

    let sourceBitmap: ImageBitmap | null = null;

    try {
      sourceBitmap = await createImageBitmap(file);
      const isStripLike =
        sourceBitmap.height <= 128 || sourceBitmap.width / Math.max(sourceBitmap.height, 1) >= 2.5;

      if (isStripLike) {
        setPreviewFromBlob(file);
        const response = await runRemoteScan(file, 'Consultando OCR remoto...', 'imagen completa');
        await applyRecognizeResponse(response);
        return;
      }

      const foregroundBounds = estimateForegroundBounds(sourceBitmap);
      const uploadCandidates = getUploadCropCandidates(
        sourceBitmap.width,
        sourceBitmap.height,
        foregroundBounds ?? undefined,
      );
      const orderedCandidateIds = [
        'foreground-bounds',
        'top-right-band',
        'bottom-left-band',
        'center-guide',
      ];
      const orderedCandidates = orderedCandidateIds
        .map((candidateId) => uploadCandidates.find((candidate) => candidate.id === candidateId))
        .filter((candidate): candidate is UploadCropCandidate => Boolean(candidate));

      let bestMatch: RecognizeResponse | null = null;

      for (let index = 0; index < orderedCandidates.length; index += 1) {
        const candidate = orderedCandidates[index];
        const croppedBlob = await createCroppedBlob(sourceBitmap, candidate.rect);
        setPreviewFromBlob(croppedBlob);

        const response = await runRemoteScan(
          croppedBlob,
          'Consultando OCR remoto...',
          `${candidate.label} (${index + 1}/${orderedCandidates.length})`,
        );

        if (response.status === 'ok' && response.serial) {
          bestMatch = bestResponse(bestMatch, response);
          if ((response.confidence ?? 0) >= ACCEPTED_CONFIDENCE_THRESHOLD) {
            break;
          }
        }
      }

      if (bestMatch) {
        await applyRecognizeResponse(bestMatch);
      } else {
        await applyRecognizeResponse({
          status: 'not_found',
          serial: null,
          raw_text: '',
          confidence: null,
          candidates: [],
          engine: 'remote',
          latency_ms: 0,
          request_id: 'local-fallback',
        });
      }
    } catch (error) {
      setResults(null);
      setScanFeedback('none');
      setCameraError(errorMessageForScan(error));
    } finally {
      sourceBitmap?.close();
      setScanProgress(null);
      setIsProcessing(false);
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

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !streamRef.current) {
      return;
    }

    if (video.srcObject !== streamRef.current) {
      video.srcObject = streamRef.current;
    }

    void video.play().catch(() => undefined);
  }, [isCameraActive]);

  useEffect(() => () => {
    stopCamera();
    revokePreviewUrl();
  }, []);

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

  useEffect(() => {
    setResults((current) => {
      if (!current || current.length === 0) {
        return current;
      }

      return current.map((result) => ({
        ...result,
        isValid: !isReportedAsInvalid(result.serialNumeric, selectedDenomination),
      }));
    });
  }, [selectedDenomination]);

  return (
    <div className="container">
      <main className="content">
        <section className="camera-card">
          <div className="camera-copy">
            <span className="camera-badge">Servidor OCR</span>
            <h2>Coloca el número de serie dentro del recuadro</h2>
            <p>
              La foto se envía a tu backend para leer solo el serial. Selecciona primero la
              denominación para validar el número contra los rangos reportados por el BCB.
            </p>
          </div>

          <div className="denomination-selector" role="group" aria-label="Seleccionar denominación">
            <div className="denomination-copy">
              <span className="denomination-label">Denominación</span>
              <small>Valor activo: Bs {selectedDenomination}</small>
            </div>
            <div className="denomination-options">
              {BILL_DENOMINATIONS.map((denomination) => (
                <button
                  key={denomination}
                  type="button"
                  className={`denomination-chip ${selectedDenomination === denomination ? 'active' : ''}`}
                  aria-pressed={selectedDenomination === denomination}
                  onClick={() => setSelectedDenomination(denomination)}
                >
                  Bs {denomination}
                </button>
              ))}
            </div>
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
              disabled={isBusy}
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
              disabled={isBusy}
              onClick={() => fileInputRef.current?.click()}
            >
              <ImageUp size={20} />
              <span>Probar con imagen</span>
            </button>

            {isCameraActive && isTorchAvailable && (
              <button
                type="button"
                className="secondary-btn"
                disabled={isBusy}
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
              <button type="button" className="secondary-btn" disabled={isBusy} onClick={stopCamera}>
                Cerrar cámara
              </button>
            )}
          </div>
        </section>

        {imageSrc && (
          <div className="preview-card">
            <img key={imageSrc} src={imageSrc} alt="Billete escaneado" className="scanned-image" />

            {isProcessing && !isCameraActive && (
              <div className="processing-overlay">
                <Loader2 className="spinner" size={48} />
                <span>Consultando OCR remoto...</span>
                <LoadingProgressBar progress={scanProgress} />
              </div>
            )}
          </div>
        )}

        {!isProcessing && results !== null && results.length > 0 && (
          <div className="results-container">
            {results.map((result) => (
              <div
                key={`${result.serialDisplay}-${result.confidence}`}
                className={`result-card ${result.isValid ? 'valid' : 'invalid'}`}
              >
                {result.isValid ? (
                  <>
                    <CheckCircle2 size={48} className="icon-valid" />
                    <h2>Billete de Bs {selectedDenomination} Válido</h2>
                    <p className="serial-code">{result.serialDisplay}</p>
                    <p>
                      No pertenece a los rangos reportados por el
                      BCB para billetes de Bs {selectedDenomination}.
                    </p>
                  </>
                ) : (
                  <>
                    <XCircle size={48} className="icon-invalid" />
                    <h2>Billete de Bs {selectedDenomination} Inválido</h2>
                    <p className="serial-code">{result.serialDisplay}</p>
                    <p>
                      ¡Cuidado! Pertenece a un lote reportado
                      robado por el BCB para billetes de Bs {selectedDenomination}.
                    </p>
                  </>
                )}
              </div>
            ))}
          </div>
        )}

        {!isProcessing && results !== null && results.length === 0 && scanFeedback === 'not-found' && (
          <div className="result-card invalid">
            <XCircle size={48} className="icon-invalid" />
            <h2>Patrón no encontrado</h2>
            <p>No se detectó un número de serie de 8 a 9 dígitos en la imagen enviada.</p>
          </div>
        )}
      </main>

      <footer className="footer-copyright">
        <button type="button" className="footer-note footer-note-trigger">
          Esta app ahora usa tu servidor OCR, así que depende de la red y del túnel activo.
        </button>
        <a href="https://github.com/nubol23" target="_blank" rel="noopener noreferrer">
          © {new Date().getFullYear()} nubol23
        </a>
      </footer>
    </div>
  );
}
