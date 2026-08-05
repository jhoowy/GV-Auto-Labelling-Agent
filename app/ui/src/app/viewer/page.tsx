"use client";
// Data Viewer (list): searchable, paginated grid of videos. Each card links to
// the per-video detail page (/viewer/[video_id]).
import Link from "next/link";
import { useEffect, useState } from "react";
import { listVideos, mediaUrl } from "../../apis/client";
import type { StatusFilter, VideoListItem } from "../../apis/types";
import { CATEGORIES, SCORE_MAX, SCORE_MIN } from "../../apis/types";
import { useAsync } from "../../lib/useAsync";
import { AsyncState, Badge, fmtTime } from "../../components/ui";

const PAGE_SIZE = 24;

// Dataset tabs; "" = all datasets. Values match metadata_json->>'dataset'.
const DATASETS: { label: string; value: string }[] = [
  { label: "All", value: "" },
  { label: "game_rating", value: "game_rating" },
  { label: "general_game_video", value: "general_game_video" },
];

// Lifecycle-status tabs; "" = all. Values match the service `status` filter.
const STATUSES: { label: string; value: StatusFilter }[] = [
  { label: "All", value: "" },
  { label: "ingested", value: "ingested" },
  { label: "labelled", value: "labelled" },
  { label: "unlabelled", value: "unlabelled" },
];

const SCORES = Array.from(
  { length: SCORE_MAX - SCORE_MIN + 1 },
  (_, i) => SCORE_MIN + i,
);

export default function ViewerList() {
  const [input, setInput] = useState("");
  const [search, setSearch] = useState("");
  const [dataset, setDataset] = useState("");
  const [status, setStatus] = useState<StatusFilter>("");
  const [page, setPage] = useState(1);

  // Advanced (category score-range) filter. `draft*` are the open-panel inputs;
  // the `applied*` values actually drive the query (Apply button commits them).
  const [advOpen, setAdvOpen] = useState(false);
  const [draftCategory, setDraftCategory] = useState("");
  const [draftMin, setDraftMin] = useState(SCORE_MIN);
  const [draftMax, setDraftMax] = useState(SCORE_MAX);
  const [category, setCategory] = useState("");
  const [scoreMin, setScoreMin] = useState(SCORE_MIN);
  const [scoreMax, setScoreMax] = useState(SCORE_MAX);

  // debounce the search box, and reset to page 1 on a new query
  useEffect(() => {
    const t = setTimeout(() => {
      setSearch(input.trim());
      setPage(1);
    }, 300);
    return () => clearTimeout(t);
  }, [input]);

  const { data, loading, error } = useAsync(
    () =>
      listVideos({
        search,
        page,
        page_size: PAGE_SIZE,
        dataset: dataset || undefined,
        status: status || undefined,
        category: category || undefined,
        // score bounds only matter when a category is chosen
        score_min: category ? scoreMin : undefined,
        score_max: category ? scoreMax : undefined,
      }),
    [search, page, dataset, status, category, scoreMin, scoreMax],
  );

  const items = data?.items ?? [];
  const total = data?.total ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="row spread">
        <h1>Data Viewer</h1>
        {total > 0 && <span className="muted small">{total} videos</span>}
      </div>

      {/* Dataset filter tabs — changing resets to page 1 */}
      <div className="row" style={{ gap: 8 }}>
        {DATASETS.map((d) => (
          <button
            key={d.value}
            className="btn"
            aria-pressed={dataset === d.value}
            style={dataset === d.value ? { fontWeight: 600, borderColor: "currentColor" } : undefined}
            onClick={() => {
              setDataset(d.value);
              setPage(1);
            }}
          >
            {d.label}
          </button>
        ))}
      </div>

      {/* Lifecycle-status filter tabs — changing resets to page 1 */}
      <div className="row" style={{ gap: 8 }}>
        {STATUSES.map((st) => (
          <button
            key={st.value || "all"}
            className="btn"
            aria-pressed={status === st.value}
            style={status === st.value ? { fontWeight: 600, borderColor: "currentColor" } : undefined}
            onClick={() => {
              setStatus(st.value);
              setPage(1);
            }}
          >
            {st.label}
          </button>
        ))}
      </div>

      <div style={{ maxWidth: 420 }}>
        <input
          type="search"
          placeholder="Search by title…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
      </div>

      {/* Collapsible advanced filter: category + score range */}
      <div className="grid" style={{ gap: 8 }}>
        <div className="row" style={{ gap: 8 }}>
          <button
            className="btn"
            aria-expanded={advOpen}
            onClick={() => setAdvOpen((o) => !o)}
          >
            {advOpen ? "▾" : "▸"} Advanced filter
          </button>
          {category && !advOpen && (
            <span className="muted small">
              {category} · score {scoreMin}–{scoreMax}
            </span>
          )}
        </div>
        {advOpen && (
          <div className="card row" style={{ gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
            <label className="grid small" style={{ gap: 4 }}>
              <span className="muted">Category</span>
              <select
                value={draftCategory}
                onChange={(e) => setDraftCategory(e.target.value)}
              >
                <option value="">Any</option>
                {CATEGORIES.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </label>
            <label className="grid small" style={{ gap: 4 }}>
              <span className="muted">Score min</span>
              <select
                value={draftMin}
                disabled={!draftCategory}
                onChange={(e) => setDraftMin(Number(e.target.value))}
              >
                {SCORES.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
            <label className="grid small" style={{ gap: 4 }}>
              <span className="muted">Score max</span>
              <select
                value={draftMax}
                disabled={!draftCategory}
                onChange={(e) => setDraftMax(Number(e.target.value))}
              >
                {SCORES.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
            <button
              className="btn"
              onClick={() => {
                // normalise the range, commit the draft, reset to page 1
                const lo = Math.min(draftMin, draftMax);
                const hi = Math.max(draftMin, draftMax);
                setCategory(draftCategory);
                setScoreMin(lo);
                setScoreMax(hi);
                setPage(1);
              }}
            >
              Apply
            </button>
            {category && (
              <button
                className="btn"
                onClick={() => {
                  setDraftCategory("");
                  setDraftMin(SCORE_MIN);
                  setDraftMax(SCORE_MAX);
                  setCategory("");
                  setScoreMin(SCORE_MIN);
                  setScoreMax(SCORE_MAX);
                  setPage(1);
                }}
              >
                Clear
              </button>
            )}
          </div>
        )}
      </div>

      <AsyncState
        loading={loading}
        error={error}
        empty={items.length === 0}
        emptyText={search ? `No videos match “${search}”.` : "No videos yet."}
      >
        <div className="grid grid-cards">
          {items.map((v) => (
            <VideoCard key={v.video_id} video={v} />
          ))}
        </div>
      </AsyncState>

      {pageCount > 1 && (
        <div className="row" style={{ justifyContent: "center", gap: 12 }}>
          <button
            className="btn"
            disabled={page <= 1 || loading}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
          >
            ← Prev
          </button>
          <span className="muted small">
            Page {page} / {pageCount}
          </span>
          <button
            className="btn"
            disabled={page >= pageCount || loading}
            onClick={() => setPage((p) => p + 1)}
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
}

function VideoCard({ video: v }: { video: VideoListItem }) {
  const [imgOk, setImgOk] = useState(true);
  const thumb = mediaUrl(v.thumbnail_url);
  return (
    <Link
      href={`/viewer/${encodeURIComponent(v.video_id)}`}
      className="card"
      style={{ display: "flex", flexDirection: "column", gap: 8, padding: 0, overflow: "hidden" }}
    >
      <div
        style={{
          position: "relative",
          width: "100%",
          aspectRatio: "16 / 9",
          background: "#e9edf2",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {thumb && imgOk ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={thumb}
            alt={v.title ?? v.video_id}
            onError={() => setImgOk(false)}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        ) : (
          <span className="muted small">no thumbnail</span>
        )}
        <span
          className="mono"
          style={{
            position: "absolute",
            right: 6,
            bottom: 6,
            padding: "1px 6px",
            borderRadius: 6,
            background: "rgba(0,0,0,0.72)",
            color: "#fff",
            fontSize: 12,
          }}
        >
          {fmtTime(v.duration_s)}
        </span>
      </div>
      <div style={{ padding: "0 12px 12px", display: "grid", gap: 4 }}>
        <strong style={{ lineHeight: 1.3 }}>{v.title ?? v.video_id}</strong>
        <div className="row spread">
          <span className="muted small mono">{v.video_id}</span>
          <Badge tone={v.status === "labelled" ? "ok" : "gray"}>{v.status}</Badge>
        </div>
        {v.n_segments != null && (
          <span className="muted small">{v.n_segments} segments</span>
        )}
      </div>
    </Link>
  );
}
