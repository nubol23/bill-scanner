CREATE TABLE IF NOT EXISTS schema_migrations (
  version TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analytics_events (
  id BIGSERIAL PRIMARY KEY,
  received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  occurred_at TIMESTAMPTZ NOT NULL,
  device_id UUID NOT NULL,
  session_id UUID NOT NULL,
  page_load_id UUID NOT NULL,
  event_name TEXT NOT NULL,
  request_id UUID NULL,
  denomination SMALLINT NULL,
  method TEXT NULL,
  outcome TEXT NULL,
  app_version TEXT NOT NULL,
  device_class TEXT NOT NULL,
  browser_family TEXT NOT NULL,
  os_family TEXT NOT NULL,
  viewport_bucket TEXT NOT NULL,
  referrer_domain TEXT NULL,
  meta JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS ocr_inferences (
  id BIGSERIAL PRIMARY KEY,
  request_id UUID NOT NULL UNIQUE,
  received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  device_id UUID NULL,
  session_id UUID NULL,
  page_load_id UUID NULL,
  denomination SMALLINT NULL,
  torch_enabled BOOLEAN NULL,
  client_started_at TIMESTAMPTZ NULL,
  outcome TEXT NOT NULL,
  engine TEXT NULL,
  latency_ms INTEGER NULL,
  raw_text TEXT NULL,
  serial TEXT NULL,
  series CHAR(1) NULL,
  confidence REAL NULL,
  candidates JSONB NOT NULL DEFAULT '[]'::jsonb,
  response_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  region_department_code TEXT NULL,
  region_department_name TEXT NULL
);

CREATE TABLE IF NOT EXISTS feedback_entries (
  id BIGSERIAL PRIMARY KEY,
  received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  device_id UUID NOT NULL,
  session_id UUID NOT NULL,
  page_load_id UUID NOT NULL,
  request_id UUID NULL REFERENCES ocr_inferences(request_id),
  rating TEXT NOT NULL,
  comment TEXT NULL,
  prompted_after_scan_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS geoip_bo_networks (
  network CIDR PRIMARY KEY,
  department_code TEXT NOT NULL,
  department_name TEXT NOT NULL,
  source TEXT NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS analytics_events_event_name_received_at_idx
  ON analytics_events (event_name, received_at DESC);

CREATE INDEX IF NOT EXISTS analytics_events_device_id_received_at_idx
  ON analytics_events (device_id, received_at DESC);

CREATE INDEX IF NOT EXISTS ocr_inferences_received_at_idx
  ON ocr_inferences (received_at DESC);

CREATE INDEX IF NOT EXISTS ocr_inferences_outcome_received_at_idx
  ON ocr_inferences (outcome, received_at DESC);

CREATE INDEX IF NOT EXISTS ocr_inferences_device_id_received_at_idx
  ON ocr_inferences (device_id, received_at DESC);

CREATE INDEX IF NOT EXISTS feedback_entries_received_at_idx
  ON feedback_entries (received_at DESC);

CREATE INDEX IF NOT EXISTS feedback_entries_device_id_received_at_idx
  ON feedback_entries (device_id, received_at DESC);
