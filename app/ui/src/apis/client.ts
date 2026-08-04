// Backend HTTP client. The UI is a thin consumer of the FastAPI surface.
// Every fetch in the app goes through here. Base URL is configurable via
// NEXT_PUBLIC_API_BASE (default http://localhost:8000).

import type {
  DbTable,
  DbTablePage,
  Label,
  Paginated,
  Policy,
  PolicyChangeRequest,
  PolicySet,
  Segment,
  TrackedSegment,
  TranslateSummary,
  Video,
  VideoListItem,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export const apiBase = () => BASE;

// Prefix a backend-relative media/asset path (e.g. thumbnail_url) with the API
// base. Absolute URLs are returned untouched.
export const mediaUrl = (path?: string | null): string | null => {
  if (!path) return null;
  return /^https?:\/\//.test(path) ? path : `${BASE}${path}`;
};

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
    ...init,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}${body ? ` — ${body}` : ""}`);
  }
  // tolerate empty bodies (204 / stub endpoints)
  const text = await res.text();
  return (text ? JSON.parse(text) : null) as T;
}

function qs(
  params: Record<string, string | number | boolean | undefined | null>,
): string {
  const parts = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
  return parts.length ? `?${parts.join("&")}` : "";
}

// ---- Videos --------------------------------------------------------------
export const listVideos = (params: {
  search?: string;
  page?: number;
  page_size?: number;
  dataset?: string;
} = {}) =>
  api<Paginated<VideoListItem>>(
    `/api/videos${qs({
      search: params.search,
      page: params.page,
      page_size: params.page_size,
      dataset: params.dataset,
    })}`,
  );

export const getVideo = (videoId: string) =>
  api<Video>(`/api/videos/${encodeURIComponent(videoId)}`);

export const listSegments = (videoId: string) =>
  api<Segment[]>(`/api/videos/${encodeURIComponent(videoId)}/segments`);

// ---- Labels --------------------------------------------------------------
export const listLabels = (segmentId?: string) =>
  api<Label[]>(
    `/api/labels${segmentId ? qs({ segment_id: segmentId }) : ""}`,
  );

// ---- Policy --------------------------------------------------------------
export const getPolicies = (category?: string) =>
  api<Policy[]>(`/api/policies${qs({ category })}`);

export const listPolicySets = () => api<PolicySet[]>(`/api/policy-sets`);

export const listChangeRequests = (status = "queued") =>
  api<PolicyChangeRequest[]>(`/api/policy-change-requests${qs({ status })}`);

// Populate Korean (presentation-only) translations for a category's nodes.
// Version-cached server-side: only changed nodes are re-translated.
export const translatePolicies = (category: string) =>
  api<TranslateSummary>(
    `/api/policies/${encodeURIComponent(category)}/translate`,
    { method: "POST" },
  );

export const resolveChangeRequest = (reqId: string, approve: boolean) =>
  api<PolicyChangeRequest | { req_id: string; approved: boolean }>(
    `/api/policy-change-requests/${encodeURIComponent(reqId)}/resolve${qs({
      approve,
    })}`,
    { method: "POST" },
  );

// ---- Node -> segment tracking --------------------------------------------
// Which segments were labelled via a decision-tree rule / attribute value.
export const getRuleSegments = (category: string, ruleIndex: number) =>
  api<TrackedSegment[]>(
    `/api/policies/${encodeURIComponent(category)}/rule/${ruleIndex}/segments`,
  );

export const getAttributeValueSegments = (
  category: string,
  name: string,
  value: string,
) =>
  api<TrackedSegment[]>(
    `/api/policies/${encodeURIComponent(category)}/attribute/${encodeURIComponent(
      name,
    )}/segments${qs({ value })}`,
  );

// ---- DB browser ----------------------------------------------------------
export const listDbTables = () => api<DbTable[]>(`/api/db/tables`);

export const getDbTable = (
  name: string,
  params: { page?: number; page_size?: number; q?: string } = {},
) =>
  api<DbTablePage>(
    `/api/db/tables/${encodeURIComponent(name)}${qs({
      page: params.page,
      page_size: params.page_size,
      q: params.q,
    })}`,
  );
