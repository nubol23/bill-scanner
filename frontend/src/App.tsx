import { useEffect, useRef, useState } from "react";
import {
  Camera,
  CheckCircle2,
  Keyboard,
  Loader2,
  XCircle,
} from "lucide-react";

import {
  ScanApiError,
  scanSerialImage,
  type RecognizeResponse,
} from "./lib/api";

const CAMERA_FRAME_WIDTH_RATIO = 0.78;
const CAMERA_FRAME_HEIGHT_RATIO = 0.18;
const CAMERA_PREVIEW_ASPECT_RATIO = 16 / 9;
const DERIVATIVE_JPEG_QUALITY = 0.92;

type BillDenomination = "10" | "20" | "50";
type ScanFeedback = "none" | "not-found";
type ActiveMethod = "none" | "camera" | "manual";

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
  series: string | null;
};

type LoadingProgressState = {
  progress: number | null;
  label: string;
  detail?: string;
};

const BILL_DENOMINATIONS = [
  "10",
  "20",
  "50",
] as const satisfies readonly BillDenomination[];

const INVALID_RANGES_BY_DENOMINATION: Record<
  BillDenomination,
  ReadonlyArray<readonly [number, number]>
> = {
  "10": [
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
  "20": [
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
  "50": [
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

function LoadingProgressBar({
  progress,
}: {
  progress: LoadingProgressState | null;
}) {
  if (!progress) {
    return null;
  }

  const normalizedProgress =
    typeof progress.progress === "number"
      ? clampProgress(progress.progress)
      : null;

  return (
    <div className="progress-panel" role="status" aria-live="polite">
      <div className="progress-meta">
        <span className="progress-label">{progress.label}</span>
        {normalizedProgress !== null && (
          <span className="progress-value">
            {Math.round(normalizedProgress)}%
          </span>
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
          <div
            className="progress-fill"
            style={{ width: `${normalizedProgress}%` }}
          />
        </div>
      )}
      {progress.detail && (
        <small className="progress-detail">{progress.detail}</small>
      )}
    </div>
  );
}

function SerialResultCard({
  result,
  denomination,
}: {
  result: SerialResult;
  denomination: BillDenomination;
}) {
  return (
    <div className={`result-card ${result.isValid ? "valid" : "invalid"}`}>
      {result.isValid ? (
        <>
          <CheckCircle2 size={48} className="icon-valid" />
          <h2>Billete de Bs {denomination} Válido</h2>
          <p className="serial-code">{result.serialDisplay}</p>
          {shouldShowSeriesWarning(result.series) && (
            <p className="series-warning-chip">
              El billete parece no ser de la serie B
            </p>
          )}
          <p>
            No pertenece a los rangos reportados por el BCB para billetes de Bs{" "}
            {denomination}.
          </p>
        </>
      ) : (
        <>
          <XCircle size={48} className="icon-invalid" />
          <h2>Billete de Bs {denomination} Inválido</h2>
          <p className="serial-code">{result.serialDisplay}</p>
          {shouldShowSeriesWarning(result.series) && (
            <p className="series-warning-chip">
              El billete parece no ser de la serie B
            </p>
          )}
          <p>
            ¡Cuidado! Pertenece a un lote reportado robado por el BCB para
            billetes de Bs {denomination}.
          </p>
        </>
      )}
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

function getPreviewViewportRect(width: number, height: number) {
  const sourceAspectRatio = width / Math.max(height, 1);

  if (Math.abs(sourceAspectRatio - CAMERA_PREVIEW_ASPECT_RATIO) < 0.01) {
    return { x: 0, y: 0, width, height };
  }

  if (sourceAspectRatio > CAMERA_PREVIEW_ASPECT_RATIO) {
    const viewportWidth = Math.max(
      1,
      Math.round(height * CAMERA_PREVIEW_ASPECT_RATIO),
    );

    return {
      x: Math.max(0, Math.round((width - viewportWidth) / 2)),
      y: 0,
      width: viewportWidth,
      height,
    };
  }

  const viewportHeight = Math.max(
    1,
    Math.round(width / CAMERA_PREVIEW_ASPECT_RATIO),
  );

  return {
    x: 0,
    y: Math.max(0, Math.round((height - viewportHeight) / 2)),
    width,
    height: viewportHeight,
  };
}

function getCaptureGuideRect(width: number, height: number) {
  const previewViewport = getPreviewViewportRect(width, height);
  const guideWidth = Math.max(
    1,
    Math.round(previewViewport.width * CAMERA_FRAME_WIDTH_RATIO),
  );
  const guideHeight = Math.max(
    1,
    Math.round(previewViewport.height * CAMERA_FRAME_HEIGHT_RATIO),
  );

  return {
    x: previewViewport.x + Math.max(0, Math.round((previewViewport.width - guideWidth) / 2)),
    y: previewViewport.y + Math.max(0, Math.round((previewViewport.height - guideHeight) / 2)),
    width: guideWidth,
    height: guideHeight,
  };
}

function releaseCanvas(canvas: HTMLCanvasElement | null) {
  if (!canvas) {
    return;
  }

  const context = canvas.getContext("2d");
  if (context) {
    context.clearRect(0, 0, canvas.width, canvas.height);
  }

  canvas.width = 0;
  canvas.height = 0;
}

function canvasToBlob(canvas: HTMLCanvasElement) {
  return new Promise<Blob | null>((resolve) => {
    canvas.toBlob(resolve, "image/jpeg", DERIVATIVE_JPEG_QUALITY);
  });
}

function extractSerialNumber(serialDisplay: string) {
  const numericText = serialDisplay.replace(/[^0-9]/g, "");
  const serialNumeric = Number(numericText);
  return Number.isFinite(serialNumeric) ? serialNumeric : null;
}

function isReportedAsInvalid(
  serialNumeric: number,
  denomination: BillDenomination,
) {
  return INVALID_RANGES_BY_DENOMINATION[denomination].some(
    ([min, max]) => serialNumeric >= min && serialNumeric <= max,
  );
}

function toSerialResult(
  serialDisplay: string,
  confidence: number,
  denomination: BillDenomination,
  series: string | null = null,
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
    series,
  };
}

function shouldShowSeriesWarning(series: string | null) {
  return Boolean(series && series !== "B");
}

function errorMessageForScan(error: unknown) {
  if (error instanceof ScanApiError) {
    switch (error.kind) {
      case "timeout":
        return error.message;
      case "unavailable":
        return error.message;
      case "rate-limited":
        return "El servicio está recibiendo muchas solicitudes. Espera unos segundos.";
      case "invalid-image":
        return error.message;
      default:
        return "No se pudo completar el análisis. Intenta de nuevo en unos segundos.";
    }
  }

  if (error instanceof Error && error.message) {
    return error.message;
  }

  return "No se pudo completar el análisis remoto.";
}

export default function App() {
  const [selectedDenomination, setSelectedDenomination] =
    useState<BillDenomination>("50");
  const [activeMethod, setActiveMethod] = useState<ActiveMethod>("none");
  const [manualSerialInput, setManualSerialInput] = useState("");
  const [manualInputError, setManualInputError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isStartingCamera, setIsStartingCamera] = useState(false);
  const [isTorchAvailable, setIsTorchAvailable] = useState(false);
  const [isTorchEnabled, setIsTorchEnabled] = useState(false);
  const [isTogglingTorch, setIsTogglingTorch] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [results, setResults] = useState<SerialResult[] | null>(null);
  const [scanFeedback, setScanFeedback] = useState<ScanFeedback>("none");
  const [scanProgress, setScanProgress] = useState<LoadingProgressState | null>(
    null,
  );

  const videoRef = useRef<HTMLVideoElement>(null);
  const manualInputRef = useRef<HTMLInputElement>(null);
  const previewUrlRef = useRef<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const isBusy = isProcessing || isStartingCamera || isTogglingTorch;
  const canClearManual =
    manualSerialInput.length > 0 || results !== null || manualInputError !== null;

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

  function clearPreview() {
    revokePreviewUrl();
    setImageSrc(null);
  }

  function clearScanOutput() {
    setResults(null);
    setScanFeedback("none");
    setManualInputError(null);
  }

  function resetFlowState() {
    clearScanOutput();
    clearPreview();
    setScanProgress(null);
    setCameraError(null);
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

    return (
      (stream.getVideoTracks()[0] as TorchCapableTrack | undefined) ?? null
    );
  }

  function supportsTorchControl(track: TorchCapableTrack | null) {
    if (!track || typeof track.getCapabilities !== "function") {
      return false;
    }

    const supportedConstraints =
      navigator.mediaDevices?.getSupportedConstraints?.() as
        | (MediaTrackSupportedConstraints & TorchSettings)
        | undefined;
    if (!supportedConstraints?.torch) {
      return false;
    }

    const capabilities = track.getCapabilities() as MediaTrackCapabilities &
      TorchSettings;
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
        advanced: [
          { torch: nextTorchState } as MediaTrackConstraintSet & TorchSettings,
        ],
      } as MediaTrackConstraints);
      setIsTorchEnabled(nextTorchState);
      setCameraError(null);
    } catch {
      setCameraError("No se pudo cambiar el flash de la cámara.");
    } finally {
      setIsTogglingTorch(false);
    }
  }

  async function applyRecognizeResponse(response: RecognizeResponse) {
    if (response.status === "ok" && response.serial) {
      const nextResult = toSerialResult(
        response.serial,
        response.confidence ?? response.candidates[0]?.confidence ?? 0.5,
        selectedDenomination,
        response.series,
      );
      setResults(nextResult ? [nextResult] : []);
      setScanFeedback("none");
      setCameraError(null);
      return;
    }

    setResults([]);
    setScanFeedback("not-found");
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
      setScanFeedback("none");
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
      setCameraError("Tu navegador no permite abrir la cámara.");
      return;
    }

    setIsStartingCamera(true);
    setCameraError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
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
      setCameraError(
        "No se pudo abrir la cámara. Revisa los permisos e inténtalo de nuevo.",
      );
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
      setCameraError(
        "La cámara todavía se está preparando. Intenta de nuevo en un segundo.",
      );
      return;
    }

    const guideRect = getCaptureGuideRect(video.videoWidth, video.videoHeight);
    const canvas = document.createElement("canvas");
    canvas.width = guideRect.width;
    canvas.height = guideRect.height;
    const context = canvas.getContext("2d", {
      alpha: false,
      willReadFrequently: true,
    });

    if (!context) {
      releaseCanvas(canvas);
      setCameraError("No se pudo preparar la captura. Intenta de nuevo.");
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
        throw new Error("No se pudo generar la captura.");
      }

      stopCamera();
      clearScanOutput();
      setPreviewFromBlob(previewBlob);
      setCameraError(null);
      await processDirectBlob(
        previewBlob,
        "Analizando imagen...",
        "captura guiada",
      );
    } catch (error) {
      setCameraError(errorMessageForScan(error));
    } finally {
      releaseCanvas(canvas);
    }
  }

  function handleManualSerialChange(
    event: React.ChangeEvent<HTMLInputElement>,
  ) {
    const sanitizedValue = event.target.value
      .replace(/[^0-9]/g, "")
      .slice(0, 9);
    setManualSerialInput(sanitizedValue);
    if (manualInputError) {
      setManualInputError(null);
    }
  }

  function handleManualSubmit() {
    if (isBusy) {
      return;
    }

    const normalizedSerial = manualSerialInput.trim();
    if (normalizedSerial.length < 8 || normalizedSerial.length > 9) {
      setManualInputError("Ingresa un número de serie de 8 a 9 dígitos.");
      return;
    }

    const manualResult = toSerialResult(
      normalizedSerial,
      1,
      selectedDenomination,
    );
    if (!manualResult) {
      setManualInputError(
        "No se pudo interpretar el número de serie ingresado.",
      );
      return;
    }

    stopCamera();
    resetFlowState();
    setResults([manualResult]);
  }

  function switchMethod(next: ActiveMethod) {
    if (isBusy || next === activeMethod) {
      return;
    }

    if (next !== "camera") {
      stopCamera();
    }

    resetFlowState();
    setActiveMethod(next);
  }

  function resetCameraFlow() {
    if (isBusy) {
      return;
    }

    resetFlowState();
    if (activeMethod !== "camera") {
      setActiveMethod("camera");
    }
    void startCamera();
  }

  function resetManualFlow() {
    if (isBusy) {
      return;
    }

    stopCamera();
    resetFlowState();
    setManualSerialInput("");
    if (activeMethod !== "manual") {
      setActiveMethod("manual");
    }
    window.requestAnimationFrame(() => manualInputRef.current?.focus());
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

  useEffect(() => {
    if (activeMethod !== "manual") {
      return;
    }

    const focusFrameId = window.requestAnimationFrame(() => {
      manualInputRef.current?.focus();
    });

    return () => window.cancelAnimationFrame(focusFrameId);
  }, [activeMethod]);

  useEffect(
    () => () => {
      stopCamera();
      revokePreviewUrl();
    },
    [],
  );

  useEffect(() => {
    if (isProcessing || (results === null && scanFeedback === "none")) {
      return;
    }

    void waitForNextPaint(2).then(() => {
      window.scrollTo({
        top: document.documentElement.scrollHeight,
        behavior: "smooth",
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
        isValid: !isReportedAsInvalid(
          result.serialNumeric,
          selectedDenomination,
        ),
      }));
    });
  }, [selectedDenomination]);

  return (
    <div className="container">
      <header className="app-header">
        <h1>Verificador de Billetes</h1>
      </header>

      <main className="content">
        {/* ── Global denomination card – always visible ── */}
        <section
          className="denomination-global-card"
          role="group"
          aria-label="Seleccionar denominación"
        >
          <div className="denomination-copy">
            <span className="denomination-label">Denominación</span>
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
        </section>

        {/* ── Method selector (initial state) ── */}
        {activeMethod === 'none' && (
          <section className="method-choice-section">
            <p className="method-choice-label">Selecciona un modo</p>
            <div className="method-choice-row">
              <button
                type="button"
                className="method-card"
                onClick={() => switchMethod('camera')}
              >
                <Camera size={32} className="method-card-icon" />
                <strong>Usar Cámara</strong>
              </button>
              <button
                type="button"
                className="method-card"
                onClick={() => switchMethod('manual')}
              >
                <Keyboard size={32} className="method-card-icon" />
                <strong>Entrada Manual</strong>
              </button>
            </div>
          </section>
        )}

        {/* ── Camera flow ── */}
        {activeMethod === 'camera' && (
          <section className="camera-container">
            <div className="section-header">
              <span className="section-title">Cámara</span>
              <div className="section-links">
                <button
                  type="button"
                  className="mode-switch-btn"
                  disabled={isBusy}
                  onClick={resetManualFlow}
                >
                  Entrada manual
                </button>
              </div>
            </div>

            {/* Scan result / error display */}
            {(imageSrc || (!isProcessing && results !== null)) ? (
              <div className="scan-output-stack">
                {imageSrc && (
                  <div className="preview-card">
                    <img
                      key={imageSrc}
                      src={imageSrc}
                      alt="Billete escaneado"
                      className="scanned-image"
                    />
                    {isProcessing && (
                      <div className="processing-overlay">
                        <Loader2 className="spinner" size={48} />
                        <span>Analizando imagen...</span>
                        <LoadingProgressBar progress={scanProgress} />
                      </div>
                    )}
                  </div>
                )}

                {!isProcessing && cameraError && results === null && (
                  <div className="error-container">
                    <XCircle size={40} className="icon-invalid" />
                    <h2>No se pudo analizar la captura</h2>
                    <p>{cameraError}</p>
                    <div className="flow-inline-actions">
                      <button
                        type="button"
                        className="primary-btn"
                        onClick={resetCameraFlow}
                      >
                        <Camera size={18} />
                        <span>Reintentar con cámara</span>
                      </button>
                      <button
                        type="button"
                        className="secondary-btn"
                        onClick={resetManualFlow}
                      >
                        <Keyboard size={18} />
                        <span>Ingresar manualmente</span>
                      </button>
                    </div>
                  </div>
                )}

                {!isProcessing && results !== null && results.length > 0 && (
                  <div className="results-container">
                    {results.map((result) => (
                      <SerialResultCard
                        key={`${result.serialDisplay}-${result.confidence}`}
                        result={result}
                        denomination={selectedDenomination}
                      />
                    ))}
                  </div>
                )}

                {/* Not-found error with recovery actions */}
                {!isProcessing &&
                  results !== null &&
                  results.length === 0 &&
                  scanFeedback === 'not-found' && (
                    <div className="error-container">
                      <XCircle size={40} className="icon-invalid" />
                      <h2>Patrón no encontrado</h2>
                      <p>
                        No se detectó un número de serie de 8 a 9 dígitos en
                        la imagen enviada.
                      </p>
                      <button
                        type="button"
                        className="primary-btn error-retry-btn"
                        onClick={resetCameraFlow}
                      >
                        <Camera size={18} />
                        <span>Reintentar con cámara</span>
                      </button>
                      <button
                        type="button"
                        className="secondary-btn"
                        onClick={resetManualFlow}
                      >
                        <Keyboard size={18} />
                        <span>Entrada manual</span>
                      </button>
                    </div>
                  )}

                {!isProcessing && results !== null && results.length > 0 && (
                  <div className="flow-inline-actions">
                    <button
                      type="button"
                      className="secondary-btn"
                      onClick={resetCameraFlow}
                    >
                      <Camera size={18} />
                      <span>Escanear otra vez</span>
                    </button>
                    <button
                      type="button"
                      className="secondary-btn"
                      onClick={resetManualFlow}
                    >
                      <Keyboard size={18} />
                      <span>Cambiar a manual</span>
                    </button>
                  </div>
                  )}

              </div>
            ) : (
              /* Camera viewport */
              <>
                <div className="camera-launch-row">
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
                          : 'Abrir cámara'}
                    </span>
                  </button>
                </div>

                {isCameraActive && (
                  <div className="camera-stage-shell">
                    <div className="camera-stage">
                      <video
                        ref={videoRef}
                        className="camera-video"
                        autoPlay
                        muted
                        playsInline
                      />
                      <div className="camera-guide" />
                      <div className="camera-guide-label">
                        Alinea el número de serie aquí
                      </div>
                    </div>
                  </div>
                )}

                {cameraError && <p className="camera-error">{cameraError}</p>}

                <div className="camera-actions">
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

              </>
            )}
          </section>
        )}

        {/* ── Manual entry flow ── */}
        {activeMethod === 'manual' && (
          <section className="manual-method-card" aria-labelledby="manual-entry-heading">
            <div className="section-header">
              <span id="manual-entry-heading" className="section-title">
                Manual
              </span>
              <div className="section-links">
                <button
                  type="button"
                  className="mode-switch-btn"
                  disabled={isBusy}
                  onClick={resetCameraFlow}
                >
                  Usar cámara
                </button>
              </div>
            </div>

            <form
              onSubmit={(event) => {
                event.preventDefault();
                handleManualSubmit();
              }}
            >
              <div className="manual-entry-field">
                <input
                  ref={manualInputRef}
                  type="text"
                  inputMode="numeric"
                  autoComplete="off"
                  className="manual-entry-input"
                  value={manualSerialInput}
                  onChange={handleManualSerialChange}
                  placeholder="Ej. 12345678"
                  aria-label="Número de serie manual"
                  aria-invalid={manualInputError ? 'true' : 'false'}
                />
                <div className="manual-entry-actions">
                  <button
                    type="submit"
                    className="primary-btn manual-submit-btn"
                    disabled={isBusy || manualSerialInput.length === 0}
                  >
                    Revisar
                  </button>
                  <button
                    type="button"
                    className="secondary-btn manual-clear-btn"
                    disabled={isBusy || !canClearManual}
                    onClick={resetManualFlow}
                  >
                    Limpiar
                  </button>
                </div>
              </div>

              {manualInputError ? (
                <p className="manual-entry-error">{manualInputError}</p>
              ) : (
                <small className="manual-entry-hint">8 a 9 dígitos</small>
              )}
            </form>

            {/* Results (manual) */}
            {!isProcessing && results !== null && results.length > 0 && (
              <div className="results-container manual-results">
                {results.map((result) => (
                  <SerialResultCard
                    key={`${result.serialDisplay}-${result.confidence}`}
                    result={result}
                    denomination={selectedDenomination}
                  />
                ))}
              </div>
            )}
          </section>
        )}

        {/* Processing spinner when no image preview (manual path) */}
        {isProcessing && !imageSrc && (
          <div className="status-box">
            <Loader2 className="spinner" size={32} />
            <LoadingProgressBar progress={scanProgress} />
          </div>
        )}
      </main>

      <footer className="footer-copyright">
        <a
          href="https://github.com/nubol23"
          target="_blank"
          rel="noopener noreferrer"
        >
          © {new Date().getFullYear()} nubol23
        </a>
      </footer>
    </div>
  );
}
