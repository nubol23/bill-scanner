import { API_BASE_URL, APP_VERSION, buildApiUrl } from './config';
import type { ClientIdentity } from './clientIdentity';

export type AnalyticsEventName =
  | 'app_opened'
  | 'denomination_selected'
  | 'method_selected'
  | 'camera_open_requested'
  | 'camera_open_result'
  | 'torch_toggled'
  | 'camera_capture_attempted'
  | 'scan_client_error'
  | 'retry_used'
  | 'fallback_used';

export type AnalyticsEventInput = {
  name: AnalyticsEventName;
  requestId?: string | null;
  denomination?: '10' | '20' | '50' | null;
  method?: 'camera' | 'manual' | null;
  outcome?: string | null;
  meta?: Record<string, unknown>;
};

type QueuedAnalyticsEvent = {
  name: AnalyticsEventName;
  occurred_at: string;
  request_id: string | null;
  denomination: '10' | '20' | '50' | null;
  method: 'camera' | 'manual' | null;
  outcome: string | null;
  meta: Record<string, unknown>;
};

type DeviceProfile = {
  deviceClass: 'mobile' | 'tablet' | 'desktop' | 'unknown';
  browserFamily: string;
  osFamily: string;
  viewportBucket: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  referrerDomain: string | null;
};

const FLUSH_INTERVAL_MS = 5_000;
const FLUSH_BATCH_SIZE = 5;
const MAX_QUEUE_LENGTH = 50;

function parseReferrerDomain() {
  if (!document.referrer) {
    return null;
  }

  try {
    return new URL(document.referrer).hostname;
  } catch {
    return null;
  }
}

function resolveViewportBucket(width: number): DeviceProfile['viewportBucket'] {
  if (width < 480) {
    return 'xs';
  }
  if (width < 768) {
    return 'sm';
  }
  if (width < 1024) {
    return 'md';
  }
  if (width < 1440) {
    return 'lg';
  }
  return 'xl';
}

function resolveBrowserFamily(userAgent: string) {
  if (/SamsungBrowser/i.test(userAgent)) {
    return 'Samsung Internet';
  }
  if (/Edg\//i.test(userAgent)) {
    return 'Edge';
  }
  if (/OPR\//i.test(userAgent)) {
    return 'Opera';
  }
  if (/Firefox\//i.test(userAgent)) {
    return 'Firefox';
  }
  if (/Chrome\//i.test(userAgent) && !/Edg\//i.test(userAgent)) {
    return 'Chrome';
  }
  if (/Safari\//i.test(userAgent) && !/Chrome\//i.test(userAgent)) {
    return 'Safari';
  }
  return 'Other';
}

function resolveOsFamily(userAgent: string) {
  if (/iPhone|iPad|iPod/i.test(userAgent)) {
    return 'iOS';
  }
  if (/Android/i.test(userAgent)) {
    return 'Android';
  }
  if (/Windows/i.test(userAgent)) {
    return 'Windows';
  }
  if (/Mac OS X|Macintosh/i.test(userAgent)) {
    return 'macOS';
  }
  if (/Linux/i.test(userAgent)) {
    return 'Linux';
  }
  return 'Unknown';
}

function resolveDeviceClass(userAgent: string): DeviceProfile['deviceClass'] {
  if (/iPad|Tablet|PlayBook|Silk/i.test(userAgent) || /Android(?!.*Mobile)/i.test(userAgent)) {
    return 'tablet';
  }
  if (/Mobi|Android|iPhone|iPod/i.test(userAgent)) {
    return 'mobile';
  }
  if (userAgent) {
    return 'desktop';
  }
  return 'unknown';
}

function getDeviceProfile(): DeviceProfile {
  const userAgent = navigator.userAgent || '';
  return {
    deviceClass: resolveDeviceClass(userAgent),
    browserFamily: resolveBrowserFamily(userAgent),
    osFamily: resolveOsFamily(userAgent),
    viewportBucket: resolveViewportBucket(window.innerWidth),
    referrerDomain: parseReferrerDomain(),
  };
}

async function postJson(url: string, payload: object) {
  await fetch(url, {
    method: 'POST',
    body: JSON.stringify(payload),
    headers: {
      'Content-Type': 'application/json',
    },
    keepalive: true,
  });
}

export class AnalyticsClient {
  private readonly identity: ClientIdentity;
  private readonly profile: DeviceProfile;
  private readonly eventsUrl: string;
  private readonly queue: QueuedAnalyticsEvent[] = [];
  private flushTimerId: number | null = null;
  private isFlushing = false;
  private readonly pageHideHandler: () => void;

  constructor(identity: ClientIdentity) {
    this.identity = identity;
    this.profile = getDeviceProfile();
    this.eventsUrl = buildApiUrl('/api/v1/events/batch');
    this.pageHideHandler = () => {
      void this.flushWithBeacon();
    };
    window.addEventListener('pagehide', this.pageHideHandler);
  }

  dispose() {
    if (this.flushTimerId !== null) {
      window.clearTimeout(this.flushTimerId);
      this.flushTimerId = null;
    }
    window.removeEventListener('pagehide', this.pageHideHandler);
    void this.flush();
  }

  track(event: AnalyticsEventInput) {
    this.queue.push({
      name: event.name,
      occurred_at: new Date().toISOString(),
      request_id: event.requestId ?? null,
      denomination: event.denomination ?? null,
      method: event.method ?? null,
      outcome: event.outcome ?? null,
      meta: event.meta ?? {},
    });

    if (this.queue.length > MAX_QUEUE_LENGTH) {
      this.queue.splice(0, this.queue.length - MAX_QUEUE_LENGTH);
    }

    if (this.queue.length >= FLUSH_BATCH_SIZE) {
      void this.flush();
      return;
    }

    if (this.flushTimerId === null) {
      this.flushTimerId = window.setTimeout(() => {
        void this.flush();
      }, FLUSH_INTERVAL_MS);
    }
  }

  async flush() {
    if (this.isFlushing || this.queue.length === 0) {
      return;
    }

    this.isFlushing = true;
    const pendingEvents = this.queue.splice(0, Math.min(this.queue.length, 10));
    if (this.flushTimerId !== null) {
      window.clearTimeout(this.flushTimerId);
      this.flushTimerId = null;
    }

    try {
      await postJson(this.eventsUrl, {
        device_id: this.identity.deviceId,
        session_id: this.identity.sessionId,
        page_load_id: this.identity.pageLoadId,
        app_version: APP_VERSION,
        device_class: this.profile.deviceClass,
        browser_family: this.profile.browserFamily,
        os_family: this.profile.osFamily,
        viewport_bucket: this.profile.viewportBucket,
        referrer_domain: this.profile.referrerDomain,
        events: pendingEvents,
      });
    } catch {
      this.queue.unshift(...pendingEvents);
      if (this.queue.length > MAX_QUEUE_LENGTH) {
        this.queue.splice(MAX_QUEUE_LENGTH);
      }
    } finally {
      this.isFlushing = false;
      if (this.queue.length > 0 && this.flushTimerId === null) {
        this.flushTimerId = window.setTimeout(() => {
          void this.flush();
        }, FLUSH_INTERVAL_MS);
      }
    }
  }

  private async flushWithBeacon() {
    if (this.queue.length === 0) {
      return;
    }

    const pendingEvents = this.queue.splice(0, Math.min(this.queue.length, 10));
    const payload = {
      device_id: this.identity.deviceId,
      session_id: this.identity.sessionId,
      page_load_id: this.identity.pageLoadId,
      app_version: APP_VERSION,
      device_class: this.profile.deviceClass,
      browser_family: this.profile.browserFamily,
      os_family: this.profile.osFamily,
      viewport_bucket: this.profile.viewportBucket,
      referrer_domain: this.profile.referrerDomain,
      events: pendingEvents,
    };

    const beaconPayload = new Blob([JSON.stringify(payload)], {
      type: 'application/json',
    });

    if (navigator.sendBeacon && navigator.sendBeacon(this.eventsUrl, beaconPayload)) {
      return;
    }

    this.queue.unshift(...pendingEvents);
    if (this.queue.length > MAX_QUEUE_LENGTH) {
      this.queue.splice(MAX_QUEUE_LENGTH);
    }
    await postJson(this.eventsUrl, payload);
  }
}

export function getEventsEndpointUrl() {
  return `${API_BASE_URL}/api/v1/events/batch`;
}
