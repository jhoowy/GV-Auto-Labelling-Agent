"use client";
// Data Viewer (detail): YouTube embed + segment timeline (colored by max label
// score) + a per-segment panel with summary, transcript, per-category labels,
// and a collapsible raw tool_trace.
import Link from "next/link";
import { useParams } from "next/navigation";
import { useMemo, useState } from "react";
import { getVideo, listLabels, listSegments } from "../../../apis/client";
import type { Label, Segment } from "../../../apis/types";
import { useAsync } from "../../../lib/useAsync";
import {
  AsyncState,
  Badge,
  JsonBlock,
  ScoreBadge,
  categoryLabel,
  fmtTime,
  scoreColor,
} from "../../../components/ui";

interface Loaded {
  segments: Segment[];
  labelsBySeg: Record<string, Label[]>;
}

// Fetch segments, then labels for each segment in parallel. Label fetches are
// tolerant: a failing/stub label endpoint yields [] rather than failing the page.
async function loadVideoData(videoId: string): Promise<Loaded> {
  const segments = (await listSegments(videoId))
    .slice()
    .sort((a, b) => a.idx - b.idx);
  const labelLists = await Promise.all(
    segments.map((s) => listLabels(s.segment_id).catch(() => [] as Label[])),
  );
  const labelsBySeg: Record<string, Label[]> = {};
  segments.forEach((s, i) => {
    labelsBySeg[s.segment_id] = labelLists[i] ?? [];
  });
  return { segments, labelsBySeg };
}

const maxScore = (labels: Label[]): number | null =>
  labels.length ? Math.max(...labels.map((l) => l.score)) : null;

export default function VideoDetail() {
  const params = useParams<{ video_id: string }>();
  const videoId = decodeURIComponent(
    Array.isArray(params.video_id) ? params.video_id[0] : params.video_id,
  );

  const video = useAsync(() => getVideo(videoId), [videoId]);
  const data = useAsync(() => loadVideoData(videoId), [videoId]);
  const [selected, setSelected] = useState<string | null>(null);

  const segments = data.data?.segments ?? [];
  const labelsBySeg = data.data?.labelsBySeg ?? {};
  const selectedSeg = segments.find((s) => s.segment_id === selected) ?? null;

  const title = video.data?.metadata?.title ?? videoId;

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div>
        <Link href="/viewer" className="small">
          ← All videos
        </Link>
        <h1 style={{ marginTop: 6 }}>{title}</h1>
        <div className="row" style={{ gap: 10 }}>
          <span className="muted small mono">{videoId}</span>
          {video.data && (
            <Badge tone={video.data.status === "labelled" ? "ok" : "gray"}>
              {video.data.status}
            </Badge>
          )}
          {video.data?.duration_s != null && (
            <span className="muted small">{fmtTime(video.data.duration_s)}</span>
          )}
        </div>
      </div>

      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        <div style={{ position: "relative", width: "100%", aspectRatio: "16 / 9" }}>
          <iframe
            title={title}
            src={`https://www.youtube.com/embed/${encodeURIComponent(videoId)}`}
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", border: 0 }}
          />
        </div>
      </div>

      {video.data?.global_overview && (
        <div className="card">
          <h3>Video overview</h3>
          <p style={{ margin: 0 }}>{video.data.global_overview}</p>
        </div>
      )}

      <section>
        <div className="row spread">
          <h2>Segment timeline</h2>
          <ScoreLegend />
        </div>
        <AsyncState
          loading={data.loading}
          error={data.error}
          empty={segments.length === 0}
          emptyText="No segments available for this video."
        >
          <Timeline
            segments={segments}
            labelsBySeg={labelsBySeg}
            selected={selected}
            onSelect={setSelected}
          />
        </AsyncState>
      </section>

      {selectedSeg && (
        <SegmentPanel
          segment={selectedSeg}
          labels={labelsBySeg[selectedSeg.segment_id] ?? []}
        />
      )}
    </div>
  );
}

function Timeline({
  segments,
  labelsBySeg,
  selected,
  onSelect,
}: {
  segments: Segment[];
  labelsBySeg: Record<string, Label[]>;
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const span = useMemo(() => {
    const start = Math.min(...segments.map((s) => s.t_start));
    const end = Math.max(...segments.map((s) => s.t_end));
    return { start, total: Math.max(1e-6, end - start) };
  }, [segments]);

  return (
    <div className="card">
      <div
        style={{
          display: "flex",
          width: "100%",
          height: 46,
          borderRadius: 8,
          overflow: "hidden",
          border: "1px solid var(--border)",
        }}
      >
        {segments.map((s) => {
          const score = maxScore(labelsBySeg[s.segment_id] ?? []);
          const width = ((s.t_end - s.t_start) / span.total) * 100;
          const isSel = s.segment_id === selected;
          return (
            <button
              key={s.segment_id}
              title={`#${s.idx} · ${fmtTime(s.t_start)}–${fmtTime(s.t_end)} · ${
                score == null ? "unlabelled" : `max score ${score}`
              }`}
              onClick={() => onSelect(s.segment_id)}
              style={{
                flex: `0 0 ${width}%`,
                minWidth: 3,
                height: "100%",
                background: scoreColor(score),
                border: "none",
                borderRight: "1px solid rgba(255,255,255,0.5)",
                outline: isSel ? "3px solid var(--accent)" : "none",
                outlineOffset: "-3px",
                cursor: "pointer",
                padding: 0,
              }}
            />
          );
        })}
      </div>
      <div className="muted small" style={{ marginTop: 6 }}>
        {segments.length} segments · click a block to inspect. Color = max label
        score across categories (grey = unlabelled).
      </div>
    </div>
  );
}

function ScoreLegend() {
  return (
    <div className="row small" style={{ gap: 6 }}>
      <span className="muted">0</span>
      {[0, 1, 2, 3, 4, 5].map((n) => (
        <span
          key={n}
          style={{
            width: 16,
            height: 12,
            borderRadius: 2,
            background: scoreColor(n),
            display: "inline-block",
          }}
        />
      ))}
      <span className="muted">5</span>
      <span
        style={{ width: 16, height: 12, borderRadius: 2, background: scoreColor(null), display: "inline-block" }}
      />
      <span className="muted">n/a</span>
    </div>
  );
}

function SegmentPanel({ segment: s, labels }: { segment: Segment; labels: Label[] }) {
  return (
    <div className="card">
      <div className="row spread">
        <h3 style={{ margin: 0 }}>
          Segment #{s.idx}{" "}
          <span className="muted small">
            {fmtTime(s.t_start)}–{fmtTime(s.t_end)}
          </span>
        </h3>
        <span className="mono muted small">{s.segment_id}</span>
      </div>

      {s.summary && (
        <div style={{ marginTop: 10 }}>
          <div className="muted small">Summary</div>
          <div>{s.summary}</div>
        </div>
      )}
      {s.transcript && (
        <div style={{ marginTop: 10 }}>
          <div className="muted small">Transcript (ASR)</div>
          <div className="small">{s.transcript}</div>
        </div>
      )}

      <hr />
      <h3>Labels by category</h3>
      {labels.length === 0 ? (
        <div className="muted small">No labels emitted for this segment yet.</div>
      ) : (
        <div className="grid" style={{ gap: 12 }}>
          {labels.map((l) => (
            <LabelCard key={l.label_id} label={l} />
          ))}
        </div>
      )}
    </div>
  );
}

function LabelCard({ label: l }: { label: Label }) {
  return (
    <div className="card" style={{ background: "var(--bg)" }}>
      <div className="row spread">
        <div className="row">
          <ScoreBadge score={l.score} />
          <strong>{categoryLabel(l.category)}</strong>
          {l.human_verified && <Badge tone="ok">verified</Badge>}
        </div>
        {l.confidence != null && (
          <span className="muted small">confidence {l.confidence.toFixed(2)}</span>
        )}
      </div>

      <p style={{ margin: "8px 0" }}>{l.rationale}</p>

      {l.cited_policy_ids?.length > 0 && (
        <div className="row" style={{ marginBottom: 6 }}>
          <span className="muted small">Cited policies:</span>
          <div className="tag-list">
            {l.cited_policy_ids.map((p) => (
              <Badge key={p}>{p}</Badge>
            ))}
          </div>
        </div>
      )}

      {l.used_segment_ids?.length > 0 && (
        <div className="row" style={{ marginBottom: 6 }}>
          <span className="muted small">Used segments:</span>
          <span className="mono small">{l.used_segment_ids.join(", ")}</span>
        </div>
      )}

      {l.evidence_attributes?.length > 0 && (
        <details>
          <summary>Evidence attributes ({l.evidence_attributes.length})</summary>
          <div className="grid" style={{ gap: 4, marginTop: 6 }}>
            {l.evidence_attributes.map((a, i) => (
              <div key={`${a.key}-${i}`} className="row small" style={{ gap: 6 }}>
                <Badge tone={a.layer === "policy" ? "default" : "gray"}>{a.layer}</Badge>
                <strong>{a.key}</strong>
                <span>= {String(a.value)}</span>
                {a.confidence != null && (
                  <span className="muted">({a.confidence.toFixed(2)})</span>
                )}
                {a.evidence && <span className="muted mono">· {a.evidence}</span>}
                <span className="muted">· {a.source}</span>
              </div>
            ))}
          </div>
        </details>
      )}

      {l.tool_trace?.length > 0 && (
        <details>
          <summary>Tool trace ({l.tool_trace.length})</summary>
          <JsonBlock value={l.tool_trace} />
        </details>
      )}
    </div>
  );
}
