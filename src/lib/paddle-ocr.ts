const DETECTION_MODEL_PATH =
  '/models/paddle-ocr/v1/ch_PP-OCRv2_det_fuse_activation/model.json';
const RECOGNITION_MODEL_PATH =
  '/models/paddle-ocr/v1/ch_PP-OCRv2_rec_fuse_activation/model.json';
const DEFAULT_MAX_DIMENSION = 1280;
const COMPACT_MAX_DIMENSION = 960;
const MOBILE_MAX_VIEWPORT_WIDTH = 768;
const OCR_RECOGNITION_TIMEOUT_MS = 8000;
const DENOMINATION_VALUES = ['200', '100', '50', '20', '10'] as const;
const DENOMINATION_SET = new Set<string>(DENOMINATION_VALUES);

type CandidateSource = 'single-token' | 'adjacent-join' | 'fallback';
type OcrPoint = [number, number];
type OcrBox = [OcrPoint, OcrPoint, OcrPoint, OcrPoint];

type OcrResponse = {
  text?: unknown;
  points?: unknown;
};

type OcrModule = {
  init: (detectionModelPath?: string, recognitionModelPath?: string) => Promise<unknown>;
  recognize: (source: OcrCompatibleCanvas) => Promise<OcrResponse | null>;
};

type SerialCandidate = {
  serialDisplay: string;
  source: CandidateSource;
  confidence: number;
  segmentIndexes: number[];
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

type RecognitionPassResult = {
  tokens: string[];
  candidates: SerialCandidate[];
};

export type ScanFeedback = 'none' | 'not-found' | 'low-confidence';
export type SerialSource = CandidateSource;
export type DetectedSerial = {
  value: string;
  confidence: number;
  source: SerialSource;
};

export type PaddleScanResult = {
  serials: DetectedSerial[];
  candidateSerials: DetectedSerial[];
  feedback: ScanFeedback;
};

const noop = () => undefined;

export class PaddleOcrTimeoutError extends Error {
  constructor(timeoutMs: number) {
    super(`El OCR tardó demasiado en responder (${timeoutMs} ms).`);
    this.name = 'PaddleOcrTimeoutError';
  }
}

let ocrModulePromise: Promise<OcrModule> | null = null;
let ocrModuleRef: OcrModule | null = null;
let ocrInitPromiseRef: Promise<void> | null = null;
let compactImageMode = false;

async function loadOcrModule() {
  if (!ocrModulePromise) {
    ocrModulePromise = import('@paddlejs-models/ocr') as Promise<OcrModule>;
  }

  try {
    return await ocrModulePromise;
  } catch (error) {
    ocrModulePromise = null;
    throw error;
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

  const sorted = [...values].sort((left, right) => left - right);
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

  const [shorter, longer] = left.length < right.length ? [left, right] : [right, left];
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
    const boxHeight =
      (distance(pointBox[0], pointBox[3]) + distance(pointBox[1], pointBox[2])) / 2;

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

function extractSingleSegmentCandidates(segments: OcrSegment[], medianNumericHeight: number) {
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

function extractAdjacentJoinCandidates(segments: OcrSegment[], medianNumericHeight: number) {
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
    if (
      (serialDisplay.length !== 8 && serialDisplay.length !== 9) ||
      DENOMINATION_SET.has(serialDisplay)
    ) {
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
      .filter(
        (segment) => !segment.isDenominationToken && hasGeometry(segment) && segment.digitOnlyText,
      )
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

function tagCanvasForOcr(canvas: HTMLCanvasElement) {
  const taggedCanvas = canvas as OcrCompatibleCanvas;
  taggedCanvas.naturalWidth = canvas.width;
  taggedCanvas.naturalHeight = canvas.height;
  return taggedCanvas;
}

function createCanvasFromSource(source: ImageBitmap) {
  const canvas = document.createElement('canvas');
  canvas.width = source.width;
  canvas.height = source.height;
  const taggedCanvas = tagCanvasForOcr(canvas);
  const context = taggedCanvas.getContext('2d', { alpha: false, willReadFrequently: true });

  if (!context) {
    releaseCanvas(taggedCanvas);
    throw new Error('No se pudo preparar la imagen para el OCR.');
  }

  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, taggedCanvas.width, taggedCanvas.height);
  context.drawImage(source, 0, 0, taggedCanvas.width, taggedCanvas.height);

  return taggedCanvas;
}

function createEnhancedCanvasVariant(sourceCanvas: OcrCompatibleCanvas): PreparedCanvas {
  const canvas = document.createElement('canvas');
  canvas.width = sourceCanvas.naturalWidth || sourceCanvas.width;
  canvas.height = sourceCanvas.naturalHeight || sourceCanvas.height;
  const taggedCanvas = tagCanvasForOcr(canvas);
  const context = taggedCanvas.getContext('2d', { alpha: false, willReadFrequently: true });

  if (!context) {
    releaseCanvas(taggedCanvas);
    return {
      canvas: sourceCanvas,
      release: noop,
    };
  }

  context.drawImage(sourceCanvas, 0, 0, taggedCanvas.width, taggedCanvas.height);
  const imageData = context.getImageData(0, 0, taggedCanvas.width, taggedCanvas.height);
  const { data } = imageData;
  let minLuminance = 255;
  let maxLuminance = 0;

  for (let index = 0; index < data.length; index += 4) {
    const luminance = data[index] * 0.299 + data[index + 1] * 0.587 + data[index + 2] * 0.114;
    minLuminance = Math.min(minLuminance, luminance);
    maxLuminance = Math.max(maxLuminance, luminance);
  }

  const dynamicRange = Math.max(1, maxLuminance - minLuminance);

  for (let index = 0; index < data.length; index += 4) {
    const luminance = data[index] * 0.299 + data[index + 1] * 0.587 + data[index + 2] * 0.114;
    const normalized = clamp((luminance - minLuminance) / dynamicRange, 0, 1);
    const contrasted = clamp((normalized - 0.5) * 2.1 + 0.5, 0, 1);
    const value = Math.round(contrasted * 255);

    data[index] = value;
    data[index + 1] = value;
    data[index + 2] = value;
    data[index + 3] = 255;
  }

  context.putImageData(imageData, 0, 0);

  return {
    canvas: taggedCanvas,
    release: () => releaseCanvas(taggedCanvas),
  };
}

function createBinaryCanvasVariant(sourceCanvas: OcrCompatibleCanvas): PreparedCanvas {
  const canvas = document.createElement('canvas');
  canvas.width = sourceCanvas.naturalWidth || sourceCanvas.width;
  canvas.height = sourceCanvas.naturalHeight || sourceCanvas.height;
  const taggedCanvas = tagCanvasForOcr(canvas);
  const context = taggedCanvas.getContext('2d', { alpha: false, willReadFrequently: true });

  if (!context) {
    releaseCanvas(taggedCanvas);
    return {
      canvas: sourceCanvas,
      release: noop,
    };
  }

  context.drawImage(sourceCanvas, 0, 0, taggedCanvas.width, taggedCanvas.height);
  const imageData = context.getImageData(0, 0, taggedCanvas.width, taggedCanvas.height);
  const { data } = imageData;
  const histogram = new Uint32Array(256);
  const luminanceValues = new Uint8Array(taggedCanvas.width * taggedCanvas.height);
  let minLuminance = 255;
  let maxLuminance = 0;

  for (let index = 0; index < luminanceValues.length; index += 1) {
    const pixelIndex = index * 4;
    const luminance = Math.round(
      data[pixelIndex] * 0.299 + data[pixelIndex + 1] * 0.587 + data[pixelIndex + 2] * 0.114,
    );
    luminanceValues[index] = luminance;
    minLuminance = Math.min(minLuminance, luminance);
    maxLuminance = Math.max(maxLuminance, luminance);
  }

  const dynamicRange = Math.max(1, maxLuminance - minLuminance);

  for (let index = 0; index < luminanceValues.length; index += 1) {
    const normalized = Math.round(((luminanceValues[index] - minLuminance) / dynamicRange) * 255);
    luminanceValues[index] = normalized;
    histogram[normalized] += 1;
  }

  let totalSum = 0;
  for (let value = 0; value < histogram.length; value += 1) {
    totalSum += value * histogram[value];
  }

  let threshold = 128;
  let maxVariance = -1;
  let backgroundWeight = 0;
  let backgroundSum = 0;

  for (let value = 0; value < histogram.length; value += 1) {
    backgroundWeight += histogram[value];
    if (backgroundWeight === 0) {
      continue;
    }

    const foregroundWeight = luminanceValues.length - backgroundWeight;
    if (foregroundWeight <= 0) {
      break;
    }

    backgroundSum += value * histogram[value];
    const backgroundMean = backgroundSum / backgroundWeight;
    const foregroundMean = (totalSum - backgroundSum) / foregroundWeight;
    const variance =
      backgroundWeight * foregroundWeight * (backgroundMean - foregroundMean) ** 2;

    if (variance > maxVariance) {
      maxVariance = variance;
      threshold = value;
    }
  }

  const adjustedThreshold = clamp(threshold, 72, 196);

  for (let index = 0; index < luminanceValues.length; index += 1) {
    const pixelIndex = index * 4;
    const value = luminanceValues[index] <= adjustedThreshold ? 0 : 255;

    data[pixelIndex] = value;
    data[pixelIndex + 1] = value;
    data[pixelIndex + 2] = value;
    data[pixelIndex + 3] = 255;
  }

  context.putImageData(imageData, 0, 0);

  return {
    canvas: taggedCanvas,
    release: () => releaseCanvas(taggedCanvas),
  };
}

async function withTimeout<T>(work: () => Promise<T>, timeoutMs: number) {
  let timeoutId: number | null = null;
  const timeoutPromise = new Promise<never>((_, reject) => {
    timeoutId = window.setTimeout(() => {
      reject(new PaddleOcrTimeoutError(timeoutMs));
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
  options?: {
    minHeight?: number;
  },
): PreparedCanvas {
  const naturalWidth = sourceCanvas.naturalWidth || sourceCanvas.width;
  const naturalHeight = sourceCanvas.naturalHeight || sourceCanvas.height;
  if (!naturalWidth || !naturalHeight) {
    return {
      canvas: sourceCanvas,
      release: noop,
    };
  }

  let targetWidth = naturalWidth;
  let targetHeight = naturalHeight;
  const longestSide = Math.max(naturalWidth, naturalHeight);

  if (longestSide > maxDimension) {
    const scale = maxDimension / longestSide;
    targetWidth = Math.max(1, Math.round(naturalWidth * scale));
    targetHeight = Math.max(1, Math.round(naturalHeight * scale));
  } else if (typeof options?.minHeight === 'number' && naturalHeight < options.minHeight) {
    const scale = options.minHeight / naturalHeight;
    targetWidth = Math.max(1, Math.round(naturalWidth * scale));
    targetHeight = Math.max(1, Math.round(naturalHeight * scale));
  }

  const needsResize = targetWidth !== naturalWidth || targetHeight !== naturalHeight;
  if (!needsResize) {
    return {
      canvas: sourceCanvas,
      release: noop,
    };
  }

  const canvas = document.createElement('canvas');
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const context = canvas.getContext('2d', { alpha: false, willReadFrequently: true });

  if (!context) {
    releaseCanvas(canvas);
    return {
      canvas: sourceCanvas,
      release: noop,
    };
  }

  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, targetWidth, targetHeight);
  context.drawImage(sourceCanvas, 0, 0, targetWidth, targetHeight);

  const resizedCanvas = tagCanvasForOcr(canvas);
  return {
    canvas: resizedCanvas,
    release: () => releaseCanvas(resizedCanvas),
  };
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

export function looksLikePaddleMemoryError(error: unknown) {
  const message =
    error instanceof Error ? error.message.toLowerCase() : String(error).toLowerCase();

  return ['memory', 'webgl', 'context', 'texture', 'quota', 'allocation'].some((hint) =>
    message.includes(hint),
  );
}

function toDetectedSerials(candidates: SerialCandidate[]) {
  return candidates.map<DetectedSerial>((candidate) => ({
    value: candidate.serialDisplay,
    confidence: Math.min(0.999, Math.max(0.01, candidate.confidence / 100)),
    source: candidate.source,
  }));
}

async function runRecognitionPass(sourceCanvas: OcrCompatibleCanvas): Promise<RecognitionPassResult> {
  const ocrModule = ocrModuleRef ?? (await ensurePaddleOcrReady());
  const response = await withTimeout(
    async () => (await ocrModule.recognize(sourceCanvas)) as OcrResponse | null,
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
}

async function runConfirmedScan(
  sourceCanvas: OcrCompatibleCanvas,
  maxDimension: number,
): Promise<PaddleScanResult> {
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

  if (recognitionPass.tokens.length === 0) {
    const upscaleHeight =
      sourceCanvas.naturalHeight < 72 ? 128 : sourceCanvas.naturalHeight < 96 ? 112 : 0;
    const retryVariants = [
      {
        createBase: () => ({
          canvas: sourceCanvas,
          release: noop,
        }),
      },
      {
        createBase: () => createEnhancedCanvasVariant(sourceCanvas),
      },
      {
        createBase: () => createBinaryCanvasVariant(sourceCanvas),
      },
    ] as const;

    for (const variant of retryVariants) {
      const baseVariant = variant.createBase();
      const preparedVariant = createPreparedCanvasVariant(baseVariant.canvas, maxDimension, {
        minHeight: upscaleHeight || undefined,
      });

      try {
        const retriedPass = await runRecognitionPass(preparedVariant.canvas);

        if (
          retriedPass.candidates.length > recognitionPass.candidates.length ||
          retriedPass.tokens.length > recognitionPass.tokens.length
        ) {
          recognitionPass = retriedPass;
        }

        if (recognitionPass.candidates.length > 0) {
          break;
        }
      } finally {
        preparedVariant.release();
        baseVariant.release();
      }
    }
  }

  const serials = toDetectedSerials(recognitionPass.candidates);
  const sawNumericHint = recognitionPass.tokens.some((token) => /[0-9OQDIL|ZSGB]/i.test(token));

  return {
    serials,
    candidateSerials: serials,
    feedback: serials.length > 0 ? 'none' : sawNumericHint ? 'low-confidence' : 'not-found',
  };
}

export async function ensurePaddleOcrReady() {
  if (ocrModuleRef) {
    return ocrModuleRef;
  }

  if (ocrInitPromiseRef) {
    await ocrInitPromiseRef;
    if (!ocrModuleRef) {
      throw new Error('OCR no disponible.');
    }
    return ocrModuleRef;
  }

  const initPromise = (async () => {
    const ocrModule = await loadOcrModule();
    await ocrModule.init(DETECTION_MODEL_PATH, RECOGNITION_MODEL_PATH);
    ocrModuleRef = ocrModule;
  })();

  ocrInitPromiseRef = initPromise;

  try {
    await initPromise;
    if (!ocrModuleRef) {
      throw new Error('OCR no disponible.');
    }

    return ocrModuleRef;
  } catch (error) {
    ocrModuleRef = null;
    throw error;
  } finally {
    if (ocrInitPromiseRef === initPromise) {
      ocrInitPromiseRef = null;
    }
  }
}

export async function scanWithPaddleOcr(source: ImageBitmap): Promise<PaddleScanResult> {
  const captureCanvas = createCanvasFromSource(source);
  const initialMaxDimension =
    compactImageMode || isMemoryConstrainedDevice() || isLikelyMobileDevice()
      ? COMPACT_MAX_DIMENSION
      : DEFAULT_MAX_DIMENSION;

  try {
    return await runConfirmedScan(captureCanvas, initialMaxDimension);
  } catch (error) {
    if (initialMaxDimension !== COMPACT_MAX_DIMENSION && looksLikePaddleMemoryError(error)) {
      compactImageMode = true;
      return runConfirmedScan(captureCanvas, COMPACT_MAX_DIMENSION);
    }

    if (looksLikePaddleMemoryError(error)) {
      compactImageMode = true;
    }

    throw error;
  } finally {
    releaseCanvas(captureCanvas);
  }
}
