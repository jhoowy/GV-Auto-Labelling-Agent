// Small presentational helpers shared across pages. No external UI deps.
import type { ReactNode } from "react";
import type { Category } from "../apis/types";
import { SCORE_MAX } from "../apis/types";

export function AsyncState({
  loading,
  error,
  empty,
  emptyText = "Nothing to show yet.",
  children,
}: {
  loading: boolean;
  error?: string | null;
  empty?: boolean;
  emptyText?: string;
  children: ReactNode;
}) {
  if (loading) return <div className="state">Loading…</div>;
  if (error)
    return (
      <div className="state error">
        Failed to load. <span className="mono">{error}</span>
      </div>
    );
  if (empty) return <div className="state">{emptyText}</div>;
  return <>{children}</>;
}

export function Badge({
  children,
  tone = "default",
}: {
  children: ReactNode;
  tone?: "default" | "gray" | "ok" | "warn" | "danger";
}) {
  const cls = tone === "default" ? "badge" : `badge ${tone}`;
  return <span className={cls}>{children}</span>;
}

// green→red across the 0..5 PEGI band. `null` => unlabelled (grey).
export function scoreColor(score: number | null | undefined): string {
  if (score == null) return "#c2c8d0";
  const t = Math.max(0, Math.min(SCORE_MAX, score)) / SCORE_MAX;
  const hue = 120 - Math.round(120 * t); // 120=green, 0=red
  return `hsl(${hue} 65% 42%)`;
}

// 0..5 PEGI-style score chip, green→red.
export function ScoreBadge({ score }: { score: number }) {
  return (
    <span className="score" style={{ background: scoreColor(score) }}>
      {score}
    </span>
  );
}

const CATEGORY_LABELS: Record<Category, string> = {
  gambling: "Gambling",
  bad_language: "Bad Language",
  sex: "Sex",
};
export const categoryLabel = (c: Category | string) =>
  CATEGORY_LABELS[c as Category] ?? c;

export function JsonBlock({ value }: { value: unknown }) {
  return <pre className="trace">{JSON.stringify(value, null, 2)}</pre>;
}

export function fmtTime(s?: number | null): string {
  if (s == null) return "—";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}
