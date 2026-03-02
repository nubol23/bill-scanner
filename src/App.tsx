import { useEffect, useRef, useState } from 'react';
import * as ocr from '@paddlejs-models/ocr';
import { Camera, CheckCircle2, Loader2, XCircle } from 'lucide-react';

const DETECTION_MODEL_PATH =
  '/models/paddle-ocr/v1/ch_PP-OCRv2_det_fuse_activation/model.json';
const RECOGNITION_MODEL_PATH =
  '/models/paddle-ocr/v1/ch_PP-OCRv2_rec_fuse_activation/model.json';
const DEFAULT_MAX_DIMENSION = 1280;
const COMPACT_MAX_DIMENSION = 960;
const MOBILE_MAX_VIEWPORT_WIDTH = 768;
const CAMERA_FRAME_WIDTH_RATIO = 0.78;
const CAMERA_FRAME_HEIGHT_RATIO = 0.18;
const DERIVATIVE_JPEG_QUALITY = 0.92;
const OCR_RECOGNITION_TIMEOUT_MS = 8000;
const DENOMINATION_VALUES = ['200', '100', '50', '20', '10'] as const;
const DENOMINATION_SET = new Set<string>(DENOMINATION_VALUES);

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
];

type OcrStatus = 'initializing' | 'ready' | 'error';
type CandidateSource = 'single-token' | 'adjacent-join' | 'fallback';
type OcrPoint = [number, number];
type OcrBox = [OcrPoint, OcrPoint, OcrPoint, OcrPoint];

type OcrResponse = {
  text?: unknown;
  points?: unknown;
};

type SerialCandidate = {
  serialDisplay: string;
  source: CandidateSource;
  confidence: number;
  segmentIndexes: number[];
};

type SerialResult = {
  isValid: boolean;
  serialDisplay: string;
  serialNumeric: number;
  source: CandidateSource;
  confidence: number;
};

type OcrSegment = {
  index: number;
  rawText: string;
  normalizedText: string;
  analysisText: string;
  pointBox: OcrBox | null;
  boxWidth: number;
  boxHeight: number;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  centerX: number;
  centerY: number;
  isLikelyVertical: boolean;
  digitOnlyText: string;
  hasTrailingLetterB: boolean;
  isStandaloneSuffixB: boolean;
  isDenominationToken: boolean;
};

type OcrCompatibleCanvas = HTMLCanvasElement & {
  naturalWidth: number;
  naturalHeight: number;
};

type PreparedCanvas = {
  canvas: OcrCompatibleCanvas;
  release: () => void;
};

type ScanFeedback = 'none' | 'not-found' | 'low-confidence';

type RecognitionPassResult = {
  tokens: string[];
  candidates: SerialCandidate[];
};

type SerialScanOutcome = {
  tokens: string[];
  results: SerialResult[];
  feedback: ScanFeedback;
};

const noop = () => undefined;

class OcrTimeoutError extends Error {
  constructor(timeoutMs: number) {
    super(`El OCR tardó demasiado en responder (${timeoutMs} ms).`);
    this.name = 'OcrTimeoutError';
  }
}

function distance(a: OcrPoint, b: OcrPoint) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function normalizeOcrSegmentText(text: string) {
  return text.toUpperCase().replace(/\s+/g, ' ').trim();
}

function sanitizeSegmentToken(text: string) {
  return text.replace(/[^A-Z0-9|]+/g, '');
}

function normalizeDigitSequence(text: string) {
  const compactText = sanitizeSegmentToken(text);
  const hasTrailingLetterB =
    compactText.endsWith('B') && /[0-9OQDIL|ZSGB]/.test(compactText.slice(0, -1));
  const numericCore = hasTrailingLetterB ? compactText.slice(0, -1) : compactText;

  const digitCore = numericCore
    .replace(/[OQD]/g, '0')
    .replace(/[IL|]/g, '1')
    .replace(/Z/g, '2')
    .replace(/S/g, '5')
    .replace(/G/g, '6')
    .replace(/B/g, '8')
    .replace(/[^0-9]/g, '');

  return {
    digitCore,
    hasTrailingLetterB,
  };
}

function parseOcrBox(rawBox: unknown): OcrBox | null {
  if (!Array.isArray(rawBox) || rawBox.length !== 4) {
    return null;
  }

  const parsedPoints = rawBox.map((point) => {
    if (!Array.isArray(point) || point.length !== 2) {
      return null;
    }

    const x = Number(point[0]);
    const y = Number(point[1]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      return null;
    }

    return [x, y] as OcrPoint;
  });

  if (parsedPoints.some((point) => point === null)) {
    return null;
  }

  return parsedPoints as OcrBox;
}

function computeMedian(values: number[]) {
  if (values.length === 0) {
    return 0;
  }

  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function getSourcePriority(source: CandidateSource) {
  switch (source) {
    case 'single-token':
      return 3;
    case 'adjacent-join':
      return 2;
    default:
      return 1;
  }
}

function isSubsequence(shorter: string, longer: string) {
  let shorterIndex = 0;

  for (const char of longer) {
    if (char === shorter[shorterIndex]) {
      shorterIndex += 1;
      if (shorterIndex === shorter.length) {
        return true;
      }
    }
  }

  return shorterIndex === shorter.length;
}

function getCandidateStrength(candidate: SerialCandidate) {
  return (
    candidate.confidence +
    candidate.serialDisplay.length * 6 +
    getSourcePriority(candidate.source) * 5
  );
}

function areSubsetDuplicateSerials(left: string, right: string) {
  if (left === right || left.length === right.length) {
    return false;
  }

  const [shorter, longer] =
    left.length < right.length ? [left, right] : [right, left];

  if (shorter.length < 7) {
    return false;
  }

  return longer.includes(shorter) || isSubsequence(shorter, longer);
}

function compareNearDuplicatePreference(current: SerialCandidate, existing: SerialCandidate) {
  if (!areSubsetDuplicateSerials(current.serialDisplay, existing.serialDisplay)) {
    return 0;
  }

  const [shorter, longer] =
    current.serialDisplay.length < existing.serialDisplay.length
      ? [current, existing]
      : [existing, current];

  if (getCandidateStrength(longer) >= getCandidateStrength(shorter) - 12) {
    return current === longer ? 1 : -1;
  }

  const currentStrength = getCandidateStrength(current);
  const existingStrength = getCandidateStrength(existing);

  if (currentStrength !== existingStrength) {
    return currentStrength > existingStrength ? 1 : -1;
  }

  if (current.serialDisplay.length !== existing.serialDisplay.length) {
    return current.serialDisplay.length > existing.serialDisplay.length ? 1 : -1;
  }

  return getSourcePriority(current.source) - getSourcePriority(existing.source);
}

function hasGeometry(segment: OcrSegment) {
  return segment.pointBox !== null && segment.boxWidth > 0 && segment.boxHeight > 0;
}

function isMemoryConstrainedDevice() {
  const deviceMemory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  return typeof deviceMemory === 'number' && deviceMemory <= 2;
}

function isLikelyMobileDevice() {
  const hasCoarsePointer =
    typeof window.matchMedia === 'function' && window.matchMedia('(pointer: coarse)').matches;
  const hasTouch = navigator.maxTouchPoints > 0;
  const hasNarrowViewport = window.innerWidth <= MOBILE_MAX_VIEWPORT_WIDTH;

  return hasCoarsePointer || (hasTouch && hasNarrowViewport);
}

function looksLikeMemoryError(error: unknown) {
  const message = error instanceof Error ? error.message.toLowerCase() : String(error).toLowerCase();
  return ['memory', 'webgl', 'context', 'texture', 'quota', 'allocation'].some((hint) =>
    message.includes(hint),
  );
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

function waitForNextPaint(frames = 1) {
  return new Promise<void>((resolve) => {
    const nextFrame = (remainingFrames: number) => {
      if (remainingFrames <= 0) {
        resolve();
        return;
      }

      window.requestAnimationFrame(() => nextFrame(remainingFrames - 1));
    };

    nextFrame(frames);
  });
}

function buildOcrSegments(tokens: string[], rawPoints: unknown) {
  const rawPointList = Array.isArray(rawPoints) ? rawPoints : [];

  return tokens.map<OcrSegment>((token, index) => {
    const normalizedText = normalizeOcrSegmentText(token);
    const analysisText = sanitizeSegmentToken(normalizedText);
    const pointBox = parseOcrBox(rawPointList[index]);
    const { digitCore, hasTrailingLetterB } = normalizeDigitSequence(analysisText);

    if (!pointBox) {
      return {
        index,
        rawText: token,
        normalizedText,
        analysisText,
        pointBox: null,
        boxWidth: 0,
        boxHeight: 0,
        minX: 0,
        maxX: 0,
        minY: 0,
        maxY: 0,
        centerX: 0,
        centerY: 0,
        isLikelyVertical: false,
        digitOnlyText: digitCore,
        hasTrailingLetterB,
        isStandaloneSuffixB: analysisText === 'B',
        isDenominationToken: analysisText === digitCore && DENOMINATION_SET.has(digitCore),
      };
    }

    const xs = pointBox.map((point) => point[0]);
    const ys = pointBox.map((point) => point[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const boxWidth = (distance(pointBox[0], pointBox[1]) + distance(pointBox[2], pointBox[3])) / 2;
    const boxHeight = (distance(pointBox[0], pointBox[3]) + distance(pointBox[1], pointBox[2])) / 2;

    return {
      index,
      rawText: token,
      normalizedText,
      analysisText,
      pointBox,
      boxWidth,
      boxHeight,
      minX,
      maxX,
      minY,
      maxY,
      centerX: (minX + maxX) / 2,
      centerY: (minY + maxY) / 2,
      isLikelyVertical: boxHeight > boxWidth * 1.25,
      digitOnlyText: digitCore,
      hasTrailingLetterB,
      isStandaloneSuffixB: analysisText === 'B',
      isDenominationToken: analysisText === digitCore && DENOMINATION_SET.has(digitCore),
    };
  });
}

function areSegmentsAdjacent(left: OcrSegment, right: OcrSegment) {
  if (!hasGeometry(left) || !hasGeometry(right)) {
    return true;
  }

  const minHeight = Math.min(left.boxHeight, right.boxHeight);
  const maxHeight = Math.max(left.boxHeight, right.boxHeight);
  if (minHeight === 0 || maxHeight / minHeight > 2.2) {
    return false;
  }

  const horizontalAligned =
    Math.abs(left.centerY - right.centerY) <= Math.max(left.boxHeight, right.boxHeight) * 0.7;
  const verticalAligned =
    Math.abs(left.centerX - right.centerX) <= Math.max(left.boxWidth, right.boxWidth) * 0.7;

  if (horizontalAligned) {
    const [earlier, later] = left.centerX <= right.centerX ? [left, right] : [right, left];
    const gap = Math.max(0, later.minX - earlier.maxX);
    return gap <= Math.max(left.boxHeight, right.boxHeight) * 1.5;
  }

  if (verticalAligned) {
    const [earlier, later] = left.centerY <= right.centerY ? [left, right] : [right, left];
    const gap = Math.max(0, later.minY - earlier.maxY);
    return gap <= Math.max(left.boxWidth, right.boxWidth) * 1.5;
  }

  return false;
}

function isStandaloneSuffixNeighbor(segment: OcrSegment, maybeSuffix: OcrSegment | undefined) {
  if (!maybeSuffix || !maybeSuffix.isStandaloneSuffixB) {
    return false;
  }

  return areSegmentsAdjacent(segment, maybeSuffix);
}

function getSuffixBoost(segment: OcrSegment, segments: OcrSegment[]) {
  let boost = 0;
  if (segment.hasTrailingLetterB) {
    boost += 12;
  }

  if (
    isStandaloneSuffixNeighbor(segment, segments[segment.index + 1]) ||
    isStandaloneSuffixNeighbor(segment, segments[segment.index - 1])
  ) {
    boost += 10;
  }

  return boost;
}

function getSegmentSizeAdjustment(segment: OcrSegment, medianNumericHeight: number) {
  if (!hasGeometry(segment) || medianNumericHeight <= 0) {
    return 0;
  }

  if (segment.boxHeight > medianNumericHeight * 1.6) {
    return -18;
  }

  if (segment.boxHeight <= medianNumericHeight * 1.15) {
    return 4;
  }

  return 0;
}

function createCandidate(
  serialDisplay: string,
  source: CandidateSource,
  segmentIndexes: number[],
  baseConfidence: number,
  extraConfidence: number,
): SerialCandidate {
  return {
    serialDisplay,
    source,
    segmentIndexes,
    confidence: clamp(baseConfidence + extraConfidence, 1, 200),
  };
}

function extractCandidatesFromPiece(
  piece: string,
  segment: OcrSegment,
  segments: OcrSegment[],
  medianNumericHeight: number,
) {
  const candidates: SerialCandidate[] = [];
  const { digitCore, hasTrailingLetterB } = normalizeDigitSequence(piece);
  const suffixBoost = getSuffixBoost(segment, segments) + (hasTrailingLetterB ? 6 : 0);
  const sizeAdjustment = getSegmentSizeAdjustment(segment, medianNumericHeight);
  const orientationAdjustment = segment.isLikelyVertical ? 2 : 0;

  if ((digitCore.length === 8 || digitCore.length === 9) && !DENOMINATION_SET.has(digitCore)) {
    const baseConfidence = digitCore.length === 9 ? 100 : 88;
    candidates.push(
      createCandidate(
        digitCore,
        'single-token',
        [segment.index],
        baseConfidence,
        suffixBoost + sizeAdjustment + orientationAdjustment,
      ),
    );
  }

  if (digitCore.length > 9) {
    for (const denomination of DENOMINATION_VALUES) {
      if (!digitCore.startsWith(denomination)) {
        continue;
      }

      const remainder = digitCore.slice(denomination.length);
      if ((remainder.length === 8 || remainder.length === 9) && !DENOMINATION_SET.has(remainder)) {
        const baseConfidence = remainder.length === 9 ? 68 : 60;
        candidates.push(
          createCandidate(
            remainder,
            'fallback',
            [segment.index],
            baseConfidence,
            suffixBoost + sizeAdjustment - 8,
          ),
        );
      }
    }
  }

  return candidates;
}

function extractSingleSegmentCandidates(
  segments: OcrSegment[],
  medianNumericHeight: number,
) {
  const candidates: SerialCandidate[] = [];

  for (const segment of segments) {
    const pieces = segment.normalizedText
      .split(/[^A-Z0-9|]+/g)
      .map(sanitizeSegmentToken)
      .filter(Boolean);

    const analysisPieces = pieces.length > 0 ? pieces : [segment.analysisText];
    const seenSerials = new Set<string>();

    for (const piece of analysisPieces) {
      for (const candidate of extractCandidatesFromPiece(
        piece,
        segment,
        segments,
        medianNumericHeight,
      )) {
        if (!seenSerials.has(candidate.serialDisplay)) {
          seenSerials.add(candidate.serialDisplay);
          candidates.push(candidate);
        }
      }
    }

    if (segment.analysisText && !analysisPieces.includes(segment.analysisText)) {
      for (const candidate of extractCandidatesFromPiece(
        segment.analysisText,
        segment,
        segments,
        medianNumericHeight,
      )) {
        if (!seenSerials.has(candidate.serialDisplay)) {
          seenSerials.add(candidate.serialDisplay);
          candidates.push(candidate);
        }
      }
    }
  }

  return candidates;
}

function extractAdjacentJoinCandidates(
  segments: OcrSegment[],
  medianNumericHeight: number,
) {
  const candidates: SerialCandidate[] = [];

  for (let index = 0; index < segments.length - 1; index += 1) {
    const left = segments[index];
    const right = segments[index + 1];

    if (
      left.isDenominationToken ||
      right.isDenominationToken ||
      left.isStandaloneSuffixB ||
      right.isStandaloneSuffixB
    ) {
      continue;
    }

    if (!left.digitOnlyText || !right.digitOnlyText) {
      continue;
    }

    if (left.digitOnlyText.length >= 8 || right.digitOnlyText.length >= 8) {
      continue;
    }

    if (!areSegmentsAdjacent(left, right)) {
      continue;
    }

    const serialDisplay = `${left.digitOnlyText}${right.digitOnlyText}`;
    if ((serialDisplay.length !== 8 && serialDisplay.length !== 9) || DENOMINATION_SET.has(serialDisplay)) {
      continue;
    }

    const suffixBoost =
      (right.hasTrailingLetterB ? 10 : 0) +
      (isStandaloneSuffixNeighbor(right, segments[index + 2]) ? 10 : 0);
    const sizeAdjustment =
      Math.round(
        (getSegmentSizeAdjustment(left, medianNumericHeight) +
          getSegmentSizeAdjustment(right, medianNumericHeight)) /
          2,
      ) + (left.isLikelyVertical || right.isLikelyVertical ? 2 : 0);

    candidates.push(
      createCandidate(
        serialDisplay,
        'adjacent-join',
        [left.index, right.index],
        serialDisplay.length === 9 ? 78 : 70,
        suffixBoost + sizeAdjustment,
      ),
    );
  }

  return candidates;
}

function extractSerialCandidates(tokens: string[], rawPoints: unknown) {
  if (tokens.length === 0) {
    return [];
  }

  const segments = buildOcrSegments(tokens, rawPoints);
  const medianNumericHeight = computeMedian(
    segments
      .filter((segment) => !segment.isDenominationToken && hasGeometry(segment) && segment.digitOnlyText)
      .map((segment) => segment.boxHeight),
  );

  const candidates = [
    ...extractSingleSegmentCandidates(segments, medianNumericHeight),
    ...extractAdjacentJoinCandidates(segments, medianNumericHeight),
  ];

  const sortedCandidates = candidates.sort((left, right) => {
    if (right.confidence !== left.confidence) {
      return right.confidence - left.confidence;
    }

    if (right.serialDisplay.length !== left.serialDisplay.length) {
      return right.serialDisplay.length - left.serialDisplay.length;
    }

    return getSourcePriority(right.source) - getSourcePriority(left.source);
  });

  const selected: SerialCandidate[] = [];
  const seenSerials = new Set<string>();
  const usedSegments = new Set<number>();

  for (const candidate of sortedCandidates) {
    if (candidate.confidence < 55 || seenSerials.has(candidate.serialDisplay)) {
      continue;
    }

    let shouldSkip = false;

    for (let index = selected.length - 1; index >= 0; index -= 1) {
      const existing = selected[index];
      const preference = compareNearDuplicatePreference(candidate, existing);

      if (preference < 0) {
        shouldSkip = true;
        break;
      }

      if (preference > 0) {
        seenSerials.delete(existing.serialDisplay);
        existing.segmentIndexes.forEach((segmentIndex) => usedSegments.delete(segmentIndex));
        selected.splice(index, 1);
      }
    }

    if (shouldSkip) {
      continue;
    }

    if (candidate.segmentIndexes.some((segmentIndex) => usedSegments.has(segmentIndex))) {
      continue;
    }

    seenSerials.add(candidate.serialDisplay);
    candidate.segmentIndexes.forEach((segmentIndex) => usedSegments.add(segmentIndex));
    selected.push(candidate);

    if (selected.length >= 6) {
      break;
    }
  }

  return selected;
}

function toSerialResults(candidates: SerialCandidate[]) {
  return candidates
    .map<SerialResult | null>((candidate) => {
      const serialNumeric = Number(candidate.serialDisplay);
      if (!Number.isFinite(serialNumeric)) {
        return null;
      }

      let isInvalid = false;
      for (const [min, max] of validRanges) {
        if (serialNumeric >= min && serialNumeric <= max) {
          isInvalid = true;
          break;
        }
      }

      return {
        isValid: !isInvalid,
        serialDisplay: candidate.serialDisplay,
        serialNumeric,
        source: candidate.source,
        confidence: candidate.confidence,
      };
    })
    .filter((result): result is SerialResult => result !== null);
}

function tagCanvasForOcr(canvas: HTMLCanvasElement) {
  const taggedCanvas = canvas as OcrCompatibleCanvas;
  taggedCanvas.naturalWidth = canvas.width;
  taggedCanvas.naturalHeight = canvas.height;
  return taggedCanvas;
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

async function withTimeout<T>(work: () => Promise<T>, timeoutMs: number) {
  let timeoutId: number | null = null;
  const timeoutPromise = new Promise<never>((_, reject) => {
    timeoutId = window.setTimeout(() => {
      reject(new OcrTimeoutError(timeoutMs));
    }, timeoutMs);
  });

  try {
    return await Promise.race([work(), timeoutPromise]);
  } finally {
    if (timeoutId !== null) {
      window.clearTimeout(timeoutId);
    }
  }
}

function createPreparedCanvasVariant(
  sourceCanvas: OcrCompatibleCanvas,
  maxDimension: number,
): PreparedCanvas {
  const naturalWidth = sourceCanvas.naturalWidth || sourceCanvas.width;
  const naturalHeight = sourceCanvas.naturalHeight || sourceCanvas.height;
  if (!naturalWidth || !naturalHeight) {
    return { canvas: sourceCanvas, release: noop };
  }

  let targetWidth = naturalWidth;
  let targetHeight = naturalHeight;
  const longestSide = Math.max(naturalWidth, naturalHeight);

  if (longestSide > maxDimension) {
    const scale = maxDimension / longestSide;
    targetWidth = Math.max(1, Math.round(naturalWidth * scale));
    targetHeight = Math.max(1, Math.round(naturalHeight * scale));
  }

  const needsResize = targetWidth !== naturalWidth || targetHeight !== naturalHeight;
  if (!needsResize) {
    return { canvas: sourceCanvas, release: noop };
  }

  const canvas = document.createElement('canvas');
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const context = canvas.getContext('2d');

  if (!context) {
    releaseCanvas(canvas);
    return { canvas: sourceCanvas, release: noop };
  }

  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, targetWidth, targetHeight);
  context.drawImage(sourceCanvas, 0, 0, targetWidth, targetHeight);

  return {
    canvas: tagCanvasForOcr(canvas),
    release: () => releaseCanvas(canvas),
  };
}

export default function App() {
  const [ocrStatus, setOcrStatus] = useState<OcrStatus>('initializing');
  const [ocrInitError, setOcrInitError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [results, setResults] = useState<SerialResult[] | null>(null);
  const [scanFeedback, setScanFeedback] = useState<ScanFeedback>('none');
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isStartingCamera, setIsStartingCamera] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const isOcrInitialized = useRef(false);
  const activeJobRef = useRef(0);
  const previewUrlRef = useRef<string | null>(null);
  const compactImageModeRef = useRef(false);
  const streamRef = useRef<MediaStream | null>(null);

  const revokePreviewUrl = () => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
  };

  const setPreviewFromBlob = (blob: Blob) => {
    revokePreviewUrl();

    const previewUrl = URL.createObjectURL(blob);
    previewUrlRef.current = previewUrl;
    setImageSrc(previewUrl);
  };

  const clearScanOutput = () => {
    setResults(null);
    setScanFeedback('none');
  };

  const stopCamera = () => {
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
  };

  const runRecognitionPass = async (
    sourceCanvas: OcrCompatibleCanvas,
  ): Promise<RecognitionPassResult> => {
    const response = await withTimeout(
      async () => (await ocr.recognize(sourceCanvas)) as OcrResponse | null,
      OCR_RECOGNITION_TIMEOUT_MS,
    );
    const tokens = Array.isArray(response?.text)
      ? response.text.filter((token): token is string => typeof token === 'string')
      : [];
    const candidates = extractSerialCandidates(tokens, response?.points);

    return {
      tokens,
      candidates,
    };
  };

  const runConfirmedScan = async (
    sourceCanvas: OcrCompatibleCanvas,
    maxDimension: number,
  ): Promise<SerialScanOutcome> => {
    const preparedCanvas = createPreparedCanvasVariant(sourceCanvas, maxDimension);
    let recognitionPass: RecognitionPassResult = {
      tokens: [],
      candidates: [],
    };

    try {
      recognitionPass = await runRecognitionPass(preparedCanvas.canvas);
    } finally {
      preparedCanvas.release();
    }

    const results = toSerialResults(recognitionPass.candidates);
    const sawNumericHint = recognitionPass.tokens.some((token) => /[0-9OQDIL|ZSGB]/i.test(token));

    return {
      tokens: recognitionPass.tokens,
      results,
      feedback: results.length > 0 ? 'none' : sawNumericHint ? 'low-confidence' : 'not-found',
    };
  };

  const processCapturedCanvas = async (captureCanvas: OcrCompatibleCanvas) => {
    const jobId = ++activeJobRef.current;
    setIsProcessing(true);

    try {
      await waitForNextPaint(2);
      if (activeJobRef.current !== jobId) {
        return;
      }

      const initialMaxDimension =
        compactImageModeRef.current || isMemoryConstrainedDevice() || isLikelyMobileDevice()
          ? COMPACT_MAX_DIMENSION
          : DEFAULT_MAX_DIMENSION;

      let scanOutcome: SerialScanOutcome;
      try {
        scanOutcome = await runConfirmedScan(captureCanvas, initialMaxDimension);
      } catch (error) {
        if (initialMaxDimension !== COMPACT_MAX_DIMENSION && looksLikeMemoryError(error)) {
          compactImageModeRef.current = true;
          scanOutcome = await runConfirmedScan(captureCanvas, COMPACT_MAX_DIMENSION);
        } else {
          throw error;
        }
      }

      if (activeJobRef.current !== jobId) {
        return;
      }

      setResults(scanOutcome.results);
      setScanFeedback(scanOutcome.feedback);
      setCameraError(null);
    } catch (error) {
      if (activeJobRef.current !== jobId) {
        return;
      }

      console.error('Error de OCR:', error);
      setResults(null);
      setScanFeedback('none');
      compactImageModeRef.current = true;

      if (error instanceof OcrTimeoutError) {
        stopCamera();
        setCameraError(
          'El OCR tardó demasiado en responder. Se cerró la cámara para liberar memoria; ábrela de nuevo e intenta otra vez.',
        );
      } else if (looksLikeMemoryError(error)) {
        stopCamera();
        setCameraError(
          'El dispositivo se quedó sin memoria durante el escaneo. Se cerró la cámara para liberar recursos.',
        );
      } else {
        setCameraError('No se pudo completar el escaneo. Intenta de nuevo.');
      }
    } finally {
      releaseCanvas(captureCanvas);

      if (activeJobRef.current === jobId) {
        setIsProcessing(false);
      }
    }
  };

  const beginScanFromCapture = (captureCanvas: OcrCompatibleCanvas, previewBlob: Blob | null) => {
    clearScanOutput();
    if (previewBlob) {
      setPreviewFromBlob(previewBlob);
    } else {
      revokePreviewUrl();
      setImageSrc(null);
    }
    setCameraError(null);
    void processCapturedCanvas(captureCanvas);
  };

  const startCamera = async () => {
    if (isProcessing || isStartingCamera) {
      return;
    }

    if (streamRef.current) {
      setCameraError(null);
      setIsCameraActive(true);
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
    } catch (error) {
      console.error('Error al abrir la cámara:', error);
      setCameraError('No se pudo abrir la cámara. Revisa los permisos e inténtalo de nuevo.');
    } finally {
      setIsStartingCamera(false);
    }
  };

  const captureCameraFrame = async () => {
    if (isProcessing) {
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
    const captureCanvas = tagCanvasForOcr(canvas);
    const context = captureCanvas.getContext('2d');

    if (!context) {
      releaseCanvas(captureCanvas);
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

    const blob = await canvasToBlob(captureCanvas);

    video.pause();
    beginScanFromCapture(captureCanvas, blob);
  };

  const initOcr = async () => {
    if (isOcrInitialized.current) {
      return;
    }

    isOcrInitialized.current = true;
    setOcrStatus('initializing');
    setOcrInitError(null);

    try {
      await ocr.init(DETECTION_MODEL_PATH, RECOGNITION_MODEL_PATH);
      setOcrStatus('ready');
    } catch (error) {
      console.error('Error al inicializar OCR:', error);
      isOcrInitialized.current = false;
      setOcrStatus('error');
      setOcrInitError(
        error instanceof Error
          ? error.message
          : 'No se pudo inicializar el motor OCR en este dispositivo.',
      );
    }
  };

  useEffect(() => {
    void initOcr();
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

  useEffect(() => {
    return () => {
      activeJobRef.current += 1;
      revokePreviewUrl();
      const stream = streamRef.current;
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <div className="container">
      <main className="content">
        {ocrStatus === 'initializing' ? (
          <div className="status-box initializing">
            <Loader2 className="spinner" size={32} />
            <p>
              Inicializando OCR
              <br />
              <small>La primera carga puede tardar unos segundos</small>
            </p>
          </div>
        ) : ocrStatus === 'error' ? (
          <div className="status-box initializing">
            <XCircle size={32} className="icon-invalid" />
            <p>
              No se pudo inicializar el OCR
              <br />
              <small>{ocrInitError ?? 'Intenta nuevamente.'}</small>
            </p>
            <button className="primary-btn" onClick={() => void initOcr()}>
              Reintentar
            </button>
          </div>
        ) : (
          <section className="camera-card">
            <div className="camera-copy">
              <span className="camera-badge">Recomendado</span>
              <h2>Captura guiada del serial</h2>
              <p>
                Coloca solo el número de serie dentro del recuadro. Puedes dejar la cámara abierta
                y seguir tomando fotos para revisar varios billetes seguidos.
              </p>
            </div>

            {isCameraActive && (
              <div className="camera-stage-shell">
                <div className="camera-stage">
                  <video
                    ref={videoRef}
                    className={`camera-video ${isProcessing && imageSrc ? 'camera-video-hidden' : ''}`}
                    autoPlay
                    muted
                    playsInline
                  />
                  {isProcessing && imageSrc && (
                    <img
                      src={imageSrc}
                      alt="Última captura del serial"
                      className="camera-still"
                    />
                  )}
                  <div className="camera-guide" />
                  <div className="camera-guide-label">Alinea el número de serie aquí</div>

                  {isProcessing && (
                    <div className="processing-overlay">
                      <Loader2 className="spinner" size={42} />
                      <span>Escaneando serial...</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {cameraError && <p className="camera-error">{cameraError}</p>}

            <div className="camera-actions">
              <button
                type="button"
                className="primary-btn"
                disabled={isProcessing || isStartingCamera}
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

              {isCameraActive && (
                <button
                  type="button"
                  className="secondary-btn"
                  disabled={isProcessing}
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
              </div>
            )}
          </div>
        )}

        {!isProcessing && results !== null && results.length > 0 && (
          <div className="results-container">
            {results.map((result) => (
              <div
                key={`${result.serialDisplay}-${result.source}`}
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
      </main>

      <footer className="footer-copyright">
        <p className="footer-note">
          Esta app corre en tu propio dispositivo y usa un modelo pequeño, así que puede
          confundirse cuando la foto está movida u oscura.
        </p>
        <a href="https://github.com/nubol23" target="_blank" rel="noopener noreferrer">
          © {new Date().getFullYear()} nubol23
        </a>
      </footer>
    </div>
  );
}
