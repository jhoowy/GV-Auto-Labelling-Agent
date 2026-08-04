"use client";
// Data Viewer (list): searchable, paginated grid of videos. Each card links to
// the per-video detail page (/viewer/[video_id]).
import Link from "next/link";
import { useEffect, useState } from "react";
import { listVideos, mediaUrl } from "../../apis/client";
import type { VideoListItem } from "../../apis/types";
import { useAsync } from "../../lib/useAsync";
import { AsyncState, Badge, fmtTime } from "../../components/ui";

const PAGE_SIZE = 24;

export default function ViewerList() {
  const [input, setInput] = useState("");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);

  // debounce the search box, and reset to page 1 on a new query
  useEffect(() => {
    const t = setTimeout(() => {
      setSearch(input.trim());
      setPage(1);
    }, 300);
    return () => clearTimeout(t);
  }, [input]);

  const { data, loading, error } = useAsync(
    () => listVideos({ search, page, page_size: PAGE_SIZE }),
    [search, page],
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

      <div style={{ maxWidth: 420 }}>
        <input
          type="search"
          placeholder="Search by title…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
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
