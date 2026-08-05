// TS mirrors of the shared Pydantic schemas (packages/schemas/schemas) plus the
// list/response envelopes the FastAPI surface returns. Kept permissive: several
// backend endpoints may still be stubs, so response types tolerate extra fields.

// Categories are dynamic strings (injected from the active policy set). The
// current PoC set is gambling / bad_language / sex, but code must not hardcode.
export type Category = string;
export type AttributeLayer = "base" | "policy";
export type PolicyType =
  | "scoring"
  | "attribute"
  | "edge_case"
  | "decision_rule";
export type VideoStatus =
  | "pending"
  | "ingesting"
  | "ingested"
  | "labelled"
  | "failed";
export type ChangeRequestStatus = "queued" | "approved" | "rejected";

export const CATEGORIES: string[] = ["gambling", "bad_language", "sex"];
export const SCORE_MIN = 0;
export const SCORE_MAX = 5;

export interface Attribute {
  key: string;
  value: string | number | boolean;
  confidence?: number | null;
  evidence?: string | null;
  layer: AttributeLayer;
  source: string;
  policy_version?: number | null;
}

export interface VideoMetadata {
  title?: string | null;
  description?: string | null;
  channel_id?: string | null;
  thumbnail_blob?: string | null;
}

export interface Video {
  video_id: string;
  metadata: VideoMetadata;
  duration_s?: number | null;
  source_blob?: string | null;
  global_overview?: string | null;
  status: VideoStatus;
}

// Lightweight row shape returned by GET /api/videos (list endpoint).
export interface VideoListItem {
  video_id: string;
  title?: string | null;
  duration_s?: number | null;
  thumbnail_url?: string | null; // path like /api/videos/{id}/thumbnail
  status: VideoStatus;
  n_segments?: number | null;
}

export interface Segment {
  segment_id: string;
  video_id: string;
  idx: number;
  t_start: number;
  t_end: number;
  clip_blob?: string | null;
  transcript?: string | null;
  summary?: string | null;
  base_attributes: Attribute[];
  status: string;
}

export interface Label {
  label_id: string;
  segment_id: string;
  category: Category;
  score: number; // 0..5
  rationale: string;
  cited_policy_ids: string[];
  evidence_attributes: Attribute[];
  used_segment_ids: string[];
  tool_trace: Record<string, unknown>[];
  confidence?: number | null;
  human_verified: boolean;
}

// DB-managed structured payloads carried on a policy node. Loosely typed —
// backends may add fields or return legacy shapes.
export interface AttributeValue {
  value: string | number | boolean;
  label?: string;
  description?: string;
  examples?: string[];
  // Concise edge-case notes disambiguating this value from neighbouring ones,
  // read by the labelling agent when assigning it. Missing/legacy payloads -> [].
  rules?: string[];
}
export interface AttributeDef {
  kind: "attribute_def";
  value_type: "categorical" | "ordinal" | "boolean";
  values?: AttributeValue[];
  guidelines?: string;
}
export interface TermLevels {
  kind: "term_levels";
  levels: Record<string, string[]>;
}
export interface DecisionRuleCond {
  attribute: string;
  op: "==" | ">=" | "<=" | "in" | "present";
  value?: unknown;
}
export interface DecisionRule {
  when: DecisionRuleCond[];
  score: number;
  note?: string;
}
export interface DecisionTree {
  kind: "decision_tree";
  default: number;
  rules: DecisionRule[];
}
// Union of the known payload shapes; kept alongside a permissive `any` on the
// Policy field so the UI can read partial/legacy payloads without narrowing.
export type StructuredData = AttributeDef | TermLevels | DecisionTree;

export interface Policy {
  policy_id: string;
  type: PolicyType;
  category: Category;
  version: number;
  parent_id?: string | null;
  text: string;
  structured_ref?: string | null;
  // DB-managed structured payload: attribute_def / term_levels / decision_tree.
  // Loosely typed (see StructuredData) so callers can read partial payloads.
  structured_data?: any;
  status: string;
  // some backends may return a pre-nested tree
  children?: Policy[];
}

// A segment that was labelled through a given policy node (decision-tree rule
// or attribute value), derived from stored label traces. See the node ->
// segment tracking endpoints.
export interface TrackedSegment {
  segment_id: string;
  video_id: string;
  score: number;
}

// Presentation-only Korean translation of a node's human-readable strings,
// stored under structured_data.i18n.ko mirroring the English field structure
// (#22). English stays authoritative; the UI falls back to English per-field
// when a Korean string is absent. `_src_version` caches the node version it was
// translated from. Loosely typed — the UI reads it defensively.
export interface PolicyI18nKo {
  _src_version?: number;
  guidelines?: string;
  values?: Record<string, { label?: string; description?: string; rules?: string[] }>;
  rules?: { note?: string }[]; // index-aligned to decision_tree rules
  default_note?: string;
}

// Summary returned by POST /api/policies/{category}/translate.
export interface TranslateSummary {
  category: string;
  translated: string[];
  skipped: string[];
  n_translated: number;
  n_skipped: number;
}

export interface PolicySet {
  version: number;
  policy_versions: Record<string, number>; // policy_id -> version
  note?: string | null;
}

export interface PolicyChangeRequest {
  req_id: string;
  proposed_change: string;
  rationale: string;
  category?: string | null;
  node_type?: string | null;
  target_policy_id?: string | null;
  affected_segments: string[];
  similar_policies: string[];
  status: ChangeRequestStatus;
}

// ---- Envelopes -----------------------------------------------------------
export interface Paginated<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

export interface DbTable {
  name: string;
  count: number;
}

// Rows may arrive as arrays aligned to `columns` or as column-keyed objects;
// the DB browser handles both.
export type DbRow = unknown[] | Record<string, unknown>;
export interface DbTablePage {
  columns: string[];
  rows: DbRow[];
  total: number;
  page: number;
  page_size: number;
}
