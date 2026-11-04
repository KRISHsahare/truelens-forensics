/**
 * The wire contract with the TrueLens backend.
 *
 * Hand-written rather than generated, and deliberately narrow: every field here mirrors a Pydantic
 * model in `backend/app/schemas`, and the comments record the semantics a renderer can get wrong.
 * The two that matter:
 *
 * - **`null` is not `0`.** A `score: null` means the check did not produce a measurement. Painting
 *   it green — the natural reading of a zero — tells a user their image passed a test that never
 *   ran, which is the one thing this product's honesty claim forbids.
 * - **absent is not `null`.** The poll is serialised with `response_model_exclude_none`, so a field
 *   that is `None` on the server is *missing from the object* rather than present-and-null. Read
 *   every optional field with `?.` / `?? fallback`, never by assuming the key exists.
 */

export type VerdictLabel = "authentic" | "ai_generated" | "edited" | "inconclusive";
export type RiskBand = "low" | "moderate" | "high" | "not_measured";
export type StageState =
  | "pending"
  | "running"
  | "done"
  | "failed"
  | "skipped"
  | "blocked";
export type JobState = "queued" | "running" | "succeeded" | "failed" | "expired";
export type TaskStatus = "ok" | "error" | "timeout" | "skipped";
export type EvidenceWeight = "strong" | "supporting" | "weak" | "unavailable";
export type OverlayKind = "heatmap" | "vector" | "spectrum" | "none";

/** The three sentences a technique shows, plus how much to trust them. */
export interface ExplanationView {
  why: string;
  found: string;
  weight: EvidenceWeight;
  weight_note: string;
}

export interface TechniqueView {
  name: string;
  label: string;
  /** Which sub-score it feeds: `ai`, `edited` or `context`. */
  group: string;
  status: TaskStatus;
  /** `null` when there is nothing to report. Never render this as a green zero. */
  score: number | null;
  band: RiskBand;
  /** The technique executed without error — not the same question as `measured`. */
  ran: boolean;
  /** It produced a finding with evidential weight. `false` ⇒ show a greyed-out card. */
  measured: boolean;
  applicable: boolean | null;
  /** Relative path (`/api/overlays/...`), signed, expiring. Prefix with the API origin. */
  overlay_url: string | null;
  overlay_kind: OverlayKind;
  explanation: ExplanationView;
  details: Record<string, unknown>;
  ms: number;
  error: string | null;
}

export interface EvidenceFlagView {
  id: string;
  kind: "deterministic" | "statistical";
  text: string;
  /** Which sub-score the flag floors: `ai`, `edited` or `overall`. */
  target: string | null;
  floor: number | null;
  forces: VerdictLabel | null;
}

export interface ReliabilityPenalty {
  id: string;
  reason: string;
  /** Multiplicative factor applied, 0..1. */
  cost: number;
  affects: string | null;
  severity: "moderate" | "severe";
}

export interface ReliabilityBlock {
  score: number;
  reasons: ReliabilityPenalty[];
  coverage: Record<string, number>;
  note: string;
}

export interface MetadataFinding {
  code: string;
  text: string;
  /** `ok` | `info` | `weak` | `moderate` | `strong`. A plain string: the analyser owns the list. */
  severity: string;
  points: number;
}

export interface MetadataBlock {
  status: string;
  present: boolean;
  exif: Record<string, unknown>;
  software: string | null;
  camera_make: string | null;
  camera_model: string | null;
  findings: MetadataFinding[];
  generator_hits: { chunk: string; keyword: string }[];
  pii_redacted: boolean;
  deterministic_flag: string | null;
  raw: Record<string, unknown>;
}

export interface C2paBlock {
  status: string;
  present: boolean;
  credential: string;
  issuer: string | null;
  claim_generator: string | null;
  assertions: Record<string, unknown>[];
  note: string;
  raw: Record<string, unknown>;
}

export interface Timings {
  total_ms: number;
  decode_ms: number;
  queue_ms: number;
  analysis_ms: number;
  ipc_ms: number;
  assembly_ms: number;
  per_technique_ms: Record<string, number>;
}

export interface AnalysisResult {
  verdict: VerdictLabel;
  ai_score: number;
  edited_score: number;
  overall: number;
  reliability: number;
  reasons: string[];
  techniques: Record<string, TechniqueView>;
  metadata: MetadataBlock;
  c2pa: C2paBlock;
  sha256: string;
  timings: Timings;
  band: RiskBand;
  headline: string;
  detail: string;
  flags: EvidenceFlagView[];
  reliability_detail: ReliabilityBlock;
  fusion: Record<string, unknown>;
  provenance: Record<string, unknown>;
  limitations: { title: string; body: string }[];
  notes: string[];
  from_cache: boolean;
  image: Record<string, unknown>;
}

export interface JobCreated {
  job_id: string;
  status: JobState;
  sha256: string;
  created_at: number;
  poll_after_ms: number;
  stages: Record<string, StageState>;
  cached: boolean;
}

export interface JobStatus {
  job_id: string;
  status: JobState;
  progress: number;
  stages: Record<string, StageState>;
  stage_ms: Record<string, number>;
  result: AnalysisResult | null;
  /** Only set when `status === "failed"`, and only present at all then. */
  error: { code: string; message: string } | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  from_cache: boolean;
}

export interface BatchItem {
  job_id: string;
  label: string;
  status: JobState;
  sha256: string;
  error: string | null;
}

export interface BatchResult {
  items: BatchItem[];
  accepted: number;
  rejected: number;
  limit: number;
}

/** The single error shape every failure leaves through. */
export interface ApiErrorBody {
  error: string;
  message: string;
  detail: string | null;
  retry_after_seconds: number | null;
  extra: Record<string, unknown> | null;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  app: string;
  version: string;
  environment: string;
  problems: string[];
  perf_profile: string;
  store: Record<string, unknown>;
  cache: Record<string, unknown>;
  concurrency: Record<string, unknown>;
  artifacts: Record<string, boolean>;
  uptime_s: number;
}

export interface TechniqueCatalogEntry {
  name: string;
  label: string;
  group: string;
  track: string;
  overlay: OverlayKind;
  heavy: boolean;
  summary: string;
}

/**
 * `evaluation` from `/api/metrics` — a pass-through of `artifacts/metrics.json`.
 *
 * Every field is optional *by contract*: the route refuses to pad a missing artifact with zeros, so
 * the page must be able to render "not measured" for any single figure. The shape is the one
 * `ml/scripts/train_cnn.py` exports; `available: false` plus `reason`/`hint`/`meaning` is what the
 * route sends when the file does not exist.
 */
export interface EvaluationMetrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  roc_auc?: number;
  pr_auc?: number;
  threshold?: number;
  balanced_accuracy?: number;
  specificity?: number;
}

export interface Evaluation {
  available: boolean;
  reason?: string;
  hint?: string;
  meaning?: string;
  /** ISO timestamp the run wrote the file. */
  trained_at?: string;
  /** Weights/optimizer identity, e.g. `efficientnet_b0_adamw_bce`. */
  model?: string;
  dataset?: {
    name?: string;
    train?: number;
    val?: number;
    test?: number;
    classes?: string[];
    sources?: { name: string; count: number }[];
    augmentations?: string[];
    note?: string;
  };
  split?: Record<string, unknown>;
  metrics?: EvaluationMetrics;
  confusion_matrix?: { labels?: string[]; matrix?: number[][] };
  /** Curve points, in the same units scikit-learn emits. Absent means no curve, not a flat line. */
  roc_curve?: { fpr?: number[]; tpr?: number[]; thresholds?: number[] };
  pr_curve?: { precision?: number[]; recall?: number[]; thresholds?: number[] };
  calibration?: { method?: string; before?: number; after?: number; note?: string };
  /** Per-class or per-augmentation breakdowns, rendered as a table when present. */
  robustness?: { name: string; value: number; note?: string }[];
  tta?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface MetricsResponse {
  detectors: Record<string, { path: string; present: boolean; enabled: boolean }>;
  fusion: {
    artifact: string;
    present: boolean;
    source: "expert" | "learned";
    calibrated: boolean;
    note: string;
    expert_weights: Record<string, number>;
  };
  techniques: TechniqueCatalogEntry[];
  evaluation: Evaluation;
}

/** Thrown by every API helper, carrying the envelope so the UI can show the server's own sentence. */
export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly detail: string | null;
  readonly retryAfterSeconds: number | null;
  readonly extra: Record<string, unknown> | null;

  constructor(status: number, body: Partial<ApiErrorBody> | null, fallbackMessage: string) {
    super(body?.message || fallbackMessage);
    this.name = "ApiError";
    this.status = status;
    this.code = body?.error || "internal";
    this.detail = body?.detail ?? null;
    this.retryAfterSeconds = body?.retry_after_seconds ?? null;
    this.extra = body?.extra ?? null;
  }
}
