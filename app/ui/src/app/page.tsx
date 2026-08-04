import { apiBase } from "../apis/client";

const CONSOLES = [
  {
    href: "/viewer",
    title: "Data Viewer",
    body: "Browse videos, then open one for a YouTube player, a shot-segment timeline colored by label severity, and per-segment labels with rationale, cited policies, and full tool trace.",
  },
  {
    href: "/policy",
    title: "Policy Manager",
    body: "Explore the versioned policy node tree per category, browse policy-set snapshots, and review the change-request queue with approve / reject actions.",
  },
  {
    href: "/db",
    title: "DB Browser",
    body: "Inspect the Postgres warehouse directly: pick a table, filter and page through its rows. Vector and JSON columns render readably.",
  },
];

export default function Home() {
  return (
    <div className="grid" style={{ gap: 20 }}>
      <div>
        <h1>Content Moderation — Operator Console</h1>
        <p className="muted">
          Agentic auto-labelling for gameplay video, judged at shot granularity.
          Goal: consistent and fully traceable labels.
        </p>
      </div>

      <div className="grid grid-cards">
        {CONSOLES.map((c) => (
          <a key={c.href} href={c.href} className="card" style={{ display: "block" }}>
            <h3>{c.title}</h3>
            <p className="muted small" style={{ margin: 0 }}>
              {c.body}
            </p>
          </a>
        ))}
      </div>

      <p className="muted small">
        Backend API base: <span className="mono">{apiBase()}</span> (set via{" "}
        <span className="mono">NEXT_PUBLIC_API_BASE</span>).
      </p>
    </div>
  );
}
