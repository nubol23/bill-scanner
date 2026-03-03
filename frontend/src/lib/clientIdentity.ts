const DEVICE_ID_KEY = 'billete.device_id';
const SESSION_ID_KEY = 'billete.session_id';
const FEEDBACK_SCAN_COUNT_KEY = 'billete.feedback.scan_count';
const FEEDBACK_NEXT_PROMPT_COUNT_KEY = 'billete.feedback.next_prompt_count';
const FEEDBACK_SUPPRESSED_UNTIL_KEY = 'billete.feedback.suppressed_until';
const FEEDBACK_INITIAL_PROMPT_COUNT = 5;
const FEEDBACK_SNOOZE_SCAN_INCREMENT = 10;
const FEEDBACK_SUPPRESS_MS = 90 * 24 * 60 * 60 * 1000;

type BrowserStorage = Pick<Storage, 'getItem' | 'setItem'>;

type StorageState = {
  deviceId: string | null;
  sessionId: string | null;
  feedbackScanCount: number;
  feedbackNextPromptCount: number;
  feedbackSuppressedUntil: number | null;
};

export type ClientIdentity = {
  deviceId: string;
  sessionId: string;
  pageLoadId: string;
};

const pageLoadId = crypto.randomUUID();
const memoryState: StorageState = {
  deviceId: null,
  sessionId: null,
  feedbackScanCount: 0,
  feedbackNextPromptCount: FEEDBACK_INITIAL_PROMPT_COUNT,
  feedbackSuppressedUntil: null,
};

function readStorageValue(storage: BrowserStorage | null, key: string) {
  if (!storage) {
    return null;
  }

  try {
    return storage.getItem(key);
  } catch {
    return null;
  }
}

function writeStorageValue(storage: BrowserStorage | null, key: string, value: string) {
  if (!storage) {
    return;
  }

  try {
    storage.setItem(key, value);
  } catch {
    void 0;
  }
}

function resolvePersistentStorage() {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function resolveSessionStorage() {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

function resolveOrCreateId(
  storage: BrowserStorage | null,
  key: string,
  fallbackKey: 'deviceId' | 'sessionId',
) {
  const existingValue = readStorageValue(storage, key);
  if (existingValue) {
    return existingValue;
  }

  if (memoryState[fallbackKey]) {
    return memoryState[fallbackKey] as string;
  }

  const nextValue = crypto.randomUUID();
  memoryState[fallbackKey] = nextValue;
  writeStorageValue(storage, key, nextValue);
  return nextValue;
}

function readNumber(storage: BrowserStorage | null, key: string, fallback: number) {
  const rawValue = readStorageValue(storage, key);
  if (!rawValue) {
    return fallback;
  }

  const parsedValue = Number(rawValue);
  return Number.isFinite(parsedValue) ? parsedValue : fallback;
}

function readNullableNumber(storage: BrowserStorage | null, key: string) {
  const rawValue = readStorageValue(storage, key);
  if (!rawValue) {
    return null;
  }

  const parsedValue = Number(rawValue);
  return Number.isFinite(parsedValue) ? parsedValue : null;
}

function writeNumber(storage: BrowserStorage | null, key: string, value: number) {
  writeStorageValue(storage, key, String(value));
}

function getFeedbackStorageState() {
  const storage = resolvePersistentStorage();
  const scanCount = storage
    ? readNumber(storage, FEEDBACK_SCAN_COUNT_KEY, 0)
    : memoryState.feedbackScanCount;
  const nextPromptCount = storage
    ? readNumber(storage, FEEDBACK_NEXT_PROMPT_COUNT_KEY, FEEDBACK_INITIAL_PROMPT_COUNT)
    : memoryState.feedbackNextPromptCount;
  const suppressedUntil = storage
    ? readNullableNumber(storage, FEEDBACK_SUPPRESSED_UNTIL_KEY)
    : memoryState.feedbackSuppressedUntil;

  return {
    storage,
    scanCount,
    nextPromptCount,
    suppressedUntil,
  };
}

function setFeedbackStorageState({
  storage,
  scanCount,
  nextPromptCount,
  suppressedUntil,
}: {
  storage: BrowserStorage | null;
  scanCount: number;
  nextPromptCount: number;
  suppressedUntil: number | null;
}) {
  memoryState.feedbackScanCount = scanCount;
  memoryState.feedbackNextPromptCount = nextPromptCount;
  memoryState.feedbackSuppressedUntil = suppressedUntil;

  if (!storage) {
    return;
  }

  writeNumber(storage, FEEDBACK_SCAN_COUNT_KEY, scanCount);
  writeNumber(storage, FEEDBACK_NEXT_PROMPT_COUNT_KEY, nextPromptCount);
  if (suppressedUntil === null) {
    writeStorageValue(storage, FEEDBACK_SUPPRESSED_UNTIL_KEY, '');
    return;
  }
  writeNumber(storage, FEEDBACK_SUPPRESSED_UNTIL_KEY, suppressedUntil);
}

export function getClientIdentity(): ClientIdentity {
  const persistentStorage = resolvePersistentStorage();
  const sessionStorage = resolveSessionStorage();

  return {
    deviceId: resolveOrCreateId(persistentStorage, DEVICE_ID_KEY, 'deviceId'),
    sessionId: resolveOrCreateId(sessionStorage, SESSION_ID_KEY, 'sessionId'),
    pageLoadId,
  };
}

export function recordQualifyingScan() {
  const state = getFeedbackStorageState();
  const nextScanCount = state.scanCount + 1;
  setFeedbackStorageState({
    storage: state.storage,
    scanCount: nextScanCount,
    nextPromptCount: state.nextPromptCount,
    suppressedUntil: state.suppressedUntil,
  });
  return nextScanCount;
}

export function getQualifyingScanCount() {
  return getFeedbackStorageState().scanCount;
}

export function shouldPromptForFeedback() {
  const { scanCount, nextPromptCount, suppressedUntil } = getFeedbackStorageState();
  if (suppressedUntil !== null && suppressedUntil > Date.now()) {
    return false;
  }

  return scanCount >= nextPromptCount;
}

export function snoozeFeedbackPrompt() {
  const state = getFeedbackStorageState();
  setFeedbackStorageState({
    storage: state.storage,
    scanCount: state.scanCount,
    nextPromptCount: state.scanCount + FEEDBACK_SNOOZE_SCAN_INCREMENT,
    suppressedUntil: state.suppressedUntil,
  });
}

export function suppressFeedbackPrompt() {
  const state = getFeedbackStorageState();
  setFeedbackStorageState({
    storage: state.storage,
    scanCount: state.scanCount,
    nextPromptCount: state.scanCount + FEEDBACK_SNOOZE_SCAN_INCREMENT,
    suppressedUntil: Date.now() + FEEDBACK_SUPPRESS_MS,
  });
}
