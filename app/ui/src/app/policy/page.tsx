"use client";
// Policy manager: versioned policy tree per category + change-request queue
// with approve / reject actions.
import { useState } from "react";
import {
  getPolicies,
  listChangeRequests,
  listPolicySets,
  resolveChangeRequest,
} from "../../apis/client";
import type { Category, Policy, PolicyChangeRequest } from "../../apis/types";
import { CATEGORIES } from "../../apis/types";
import { useAsync } from "../../lib/useAsync";
import { AsyncState, Badge, categoryLabel } from "../../components/ui";

export default function PolicyManager() {
  const [category, setCategory] = useState<Category>("gambling");
  return (
    <div className="grid" style={{ gap: 20 }}>
      <div className="row spread">
        <h1>Policy Manager</h1>
        <div className="row">
          <span className="muted small">Category</span>
          <select
            style={{ width: 180 }}
            value={category}
            onChange={(e) => setCategory(e.target.value as Category)}
          >
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>
                {categoryLabel(c)}
              </option>
            ))}
          </select>
        </div>
      </div>
      <PolicyTreeSection category={category} />
      <PolicySetSection />
      <QueueSection />
    </div>
  );
}

function PolicySetSection() {
  const { data, loading, error } = useAsync(() => listPolicySets(), []);
  const sets = data ?? [];
  return (
    <section>
      <h2>Policy-set versions</h2>
      <AsyncState
        loading={loading}
        error={error}
        empty={sets.length === 0}
        emptyText="No policy sets snapshotted yet."
      >
        <div className="grid" style={{ gap: 8 }}>
          {sets.map((ps) => {
            const pins = Object.entries(ps.policy_versions ?? {});
            return (
              <div key={ps.version} className="card">
                <div className="row spread">
                  <strong>policy-set v{ps.version}</strong>
                  <span className="muted small">{pins.length} pinned nodes</span>
                </div>
                {ps.note && <p className="small" style={{ margin: "6px 0 0" }}>{ps.note}</p>}
                {pins.length > 0 && (
                  <details style={{ marginTop: 6 }}>
                    <summary>Pinned versions</summary>
                    <div className="tag-list" style={{ marginTop: 6 }}>
                      {pins.map(([pid, v]) => (
                        <Badge key={pid} tone="gray">
                          {pid} v{v}
                        </Badge>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            );
          })}
        </div>
      </AsyncState>
    </section>
  );
}

const TYPE_TONE: Record<string, "default" | "gray" | "warn"> = {
  scoring: "default",
  attribute: "gray",
  edge_case: "warn",
};

const TYPE_GROUPS: { type: string; label: string }[] = [
  { type: "scoring", label: "Scoring rubric" },
  { type: "attribute", label: "Attribute definitions" },
  { type: "edge_case", label: "Edge-case rules" },
];

function PolicyTreeSection({ category }: { category: Category }) {
  const { data, loading, error } = useAsync(() => getPolicies(category), [category]);
  const roots = normalizeTree(data ?? []);
  return (
    <section>
      <h2>Policy tree — {categoryLabel(category)}</h2>
      <AsyncState
        loading={loading}
        error={error}
        empty={roots.length === 0}
        emptyText="No policy nodes for this category."
      >
        <div className="grid" style={{ gap: 16 }}>
          {TYPE_GROUPS.map((g) => {
            const groupNodes = roots.filter((n) => n.type === g.type);
            if (groupNodes.length === 0) return null;
            return (
              <div key={g.type}>
                <h3>{g.label}</h3>
                <div className="grid" style={{ gap: 8 }}>
                  {groupNodes.map((n) => (
                    <PolicyNode key={n.policy_id} node={n} depth={0} />
                  ))}
                </div>
              </div>
            );
          })}
          {/* any roots whose type isn't a known group */}
          {roots
            .filter((n) => !TYPE_GROUPS.some((g) => g.type === n.type))
            .map((n) => (
              <PolicyNode key={n.policy_id} node={n} depth={0} />
            ))}
        </div>
      </AsyncState>
    </section>
  );
}

function PolicyNode({ node, depth }: { node: Policy; depth: number }) {
  return (
    <div style={{ marginLeft: depth * 20 }}>
      <div className="card">
        <div className="row spread">
          <div className="row">
            <Badge tone={TYPE_TONE[node.type] ?? "gray"}>{node.type}</Badge>
            <span className="mono small">{node.policy_id}</span>
            <span className="muted small">v{node.version}</span>
          </div>
          <Badge tone={node.status === "active" ? "ok" : "gray"}>{node.status}</Badge>
        </div>
        <p style={{ margin: "8px 0 0" }}>{node.text}</p>
        {node.structured_ref && (
          <div className="muted small mono">structured_ref: {node.structured_ref}</div>
        )}
      </div>
      {node.children?.map((c) => (
        <div key={c.policy_id} style={{ marginTop: 8 }}>
          <PolicyNode node={c} depth={depth + 1} />
        </div>
      ))}
    </div>
  );
}

function QueueSection() {
  const { data, loading, error, reload } = useAsync(() => listChangeRequests("queued"), []);
  const reqs = data ?? [];
  const [busy, setBusy] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);

  async function resolve(reqId: string, approve: boolean) {
    setBusy(reqId);
    setMsg(null);
    try {
      await resolveChangeRequest(reqId, approve);
      setMsg(`Request ${reqId} ${approve ? "approved" : "rejected"}.`);
      reload();
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  return (
    <section>
      <div className="row spread">
        <h2>Change-request queue</h2>
        {reqs.length > 0 && <Badge tone="warn">{reqs.length} pending</Badge>}
      </div>
      {msg && <div className="small muted" style={{ marginBottom: 8 }}>{msg}</div>}
      <AsyncState
        loading={loading}
        error={error}
        empty={reqs.length === 0}
        emptyText="No queued change requests."
      >
        <div className="grid" style={{ gap: 12 }}>
          {reqs.map((r) => (
            <ChangeRequestCard
              key={r.req_id}
              req={r}
              busy={busy === r.req_id}
              onResolve={resolve}
            />
          ))}
        </div>
      </AsyncState>
    </section>
  );
}

function ChangeRequestCard({
  req,
  busy,
  onResolve,
}: {
  req: PolicyChangeRequest;
  busy: boolean;
  onResolve: (reqId: string, approve: boolean) => void;
}) {
  return (
    <div className="card">
      <div className="row spread">
        <span className="mono small">{req.req_id}</span>
        <Badge tone="gray">{req.status}</Badge>
      </div>
      <div style={{ marginTop: 8 }}>
        <div className="muted small">Proposed change</div>
        <p style={{ margin: "2px 0" }}>{req.proposed_change}</p>
      </div>
      <div style={{ marginTop: 4 }}>
        <div className="muted small">Rationale</div>
        <p style={{ margin: "2px 0" }}>{req.rationale}</p>
      </div>
      {req.affected_segments?.length > 0 && (
        <div className="small muted">
          Affected segments: <span className="mono">{req.affected_segments.join(", ")}</span>
        </div>
      )}
      {req.similar_policies?.length > 0 && (
        <div className="small muted">
          Similar policies: <span className="mono">{req.similar_policies.join(", ")}</span>
        </div>
      )}
      <div className="row" style={{ marginTop: 10 }}>
        <button className="btn ok" disabled={busy} onClick={() => onResolve(req.req_id, true)}>
          Approve
        </button>
        <button className="btn danger" disabled={busy} onClick={() => onResolve(req.req_id, false)}>
          Reject
        </button>
      </div>
    </div>
  );
}

// Accepts either a flat Policy[] (linked by parent_id) or an already-nested
// tree, and returns root nodes with `children` populated.
function normalizeTree(data: Policy[]): Policy[] {
  if (!Array.isArray(data) || data.length === 0) return [];
  const alreadyNested = data.some((p) => p.children && p.children.length > 0);
  if (alreadyNested) {
    const roots = data.filter((p) => !p.parent_id);
    return roots.length ? roots : data;
  }

  const byId = new Map<string, Policy>();
  data.forEach((p) => byId.set(p.policy_id, { ...p, children: [] }));
  const roots: Policy[] = [];
  byId.forEach((p) => {
    if (p.parent_id && byId.has(p.parent_id)) {
      byId.get(p.parent_id)!.children!.push(p);
    } else {
      roots.push(p);
    }
  });
  return roots;
}
