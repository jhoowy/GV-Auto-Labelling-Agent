"use client";
// Policy manager: versioned policy tree per category + change-request queue
// with approve / reject actions.
import { useEffect, useMemo, useState } from "react";
import {
  getAttributeValueSegments,
  getPolicies,
  getRuleSegments,
  listChangeRequests,
  listPolicySets,
  resolveChangeRequest,
} from "../../apis/client";
import type {
  Category,
  Policy,
  PolicyChangeRequest,
  TrackedSegment,
} from "../../apis/types";
import { CATEGORIES } from "../../apis/types";
import { useAsync } from "../../lib/useAsync";
import {
  AsyncState,
  Badge,
  ScoreBadge,
  categoryLabel,
  scoreColor,
} from "../../components/ui";

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
  decision_rule: "default",
};

const TYPE_GROUPS: { type: string; label: string }[] = [
  { type: "scoring", label: "Scoring rubric" },
  { type: "attribute", label: "Attribute definitions" },
  { type: "edge_case", label: "Edge-case rules" },
];

// The bare attribute name a decision-rule condition references. Attribute nodes
// are stored as `{cat}.attr.{name}`; conditions use just `{name}`.
function attrKey(node: Policy): string {
  const parts = node.policy_id.split(".attr.");
  return parts.length > 1 ? parts[1] : node.policy_id;
}

function ruleAttrs(rule: any): string[] {
  const conds: any[] = Array.isArray(rule?.when) ? rule.when : [];
  return conds.map((c) => String(c?.attribute)).filter(Boolean);
}

// Mirrors the aggregator cap (tools/tracking.RESULT_CAP): a full page means
// there may be more matches than shown.
const TRACK_CAP = 200;

// Node -> segment tracking: fetches the segments labelled via a policy node
// (a decision-tree rule or an attribute value) and lists them, each linking to
// its video in the viewer. Remount (via a `key`) to refetch for a new node.
function SegmentTrackPanel({
  title,
  fetcher,
  onClose,
}: {
  title: string;
  fetcher: () => Promise<TrackedSegment[]>;
  onClose: () => void;
}) {
  const { data, loading, error } = useAsync(fetcher, []);
  const segs = data ?? [];
  return (
    <div className="card" style={{ marginTop: 8 }}>
      <div className="row spread">
        <strong className="small">{title}</strong>
        <button className="btn" onClick={onClose}>
          Close
        </button>
      </div>
      <AsyncState
        loading={loading}
        error={error}
        empty={segs.length === 0}
        emptyText="No segments were labelled via this node."
      >
        <div className="grid" style={{ gap: 4, marginTop: 6 }}>
          {segs.map((s) => (
            <div key={s.segment_id} className="row spread small">
              <a className="mono" href={`/viewer/${encodeURIComponent(s.video_id)}`}>
                {s.segment_id}
              </a>
              <ScoreBadge score={s.score} />
            </div>
          ))}
        </div>
        {segs.length >= TRACK_CAP && (
          <div className="muted small" style={{ marginTop: 6 }}>
            Showing first {TRACK_CAP} matches.
          </div>
        )}
      </AsyncState>
    </div>
  );
}

function PolicyTreeSection({ category }: { category: Category }) {
  const { data, loading, error } = useAsync(() => getPolicies(category), [category]);
  const flat = data ?? [];
  const roots = normalizeTree(flat);
  const decisionNode = flat.find((n) => n.type === "decision_rule") ?? null;
  const rules: any[] = Array.isArray(decisionNode?.structured_data?.rules)
    ? decisionNode!.structured_data.rules
    : [];

  // Clicking a decision node selects a rule; its condition attributes light up
  // the matching attribute dictionary cards.
  const [selectedRule, setSelectedRule] = useState<number | null>(null);
  useEffect(() => setSelectedRule(null), [category]);
  const highlighted = useMemo(
    () =>
      new Set<string>(
        selectedRule != null && rules[selectedRule]
          ? ruleAttrs(rules[selectedRule])
          : [],
      ),
    [selectedRule, rules],
  );

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
                    <PolicyNode
                      key={n.policy_id}
                      node={n}
                      depth={0}
                      highlighted={highlighted}
                      category={category}
                    />
                  ))}
                </div>
              </div>
            );
          })}
          {/* any roots whose type isn't a known group */}
          {roots
            .filter((n) => !TYPE_GROUPS.some((g) => g.type === n.type))
            .map((n) => (
              <PolicyNode
                key={n.policy_id}
                node={n}
                depth={0}
                highlighted={highlighted}
                category={category}
              />
            ))}
        </div>
        <DecisionTreeDiagram
          node={decisionNode}
          selected={selectedRule}
          onSelect={(i) => setSelectedRule((cur) => (cur === i ? null : i))}
        />
        {/* Selecting a rule also lists the segments it labelled. */}
        {selectedRule != null && (
          <SegmentTrackPanel
            key={`${category}-rule-${selectedRule}`}
            title={`Segments labelled via rule #${selectedRule + 1}`}
            fetcher={() => getRuleSegments(category, selectedRule)}
            onClose={() => setSelectedRule(null)}
          />
        )}
      </AsyncState>
    </section>
  );
}

// ── Attribute dictionary ──────────────────────────────────────────────────
// A readable per-attribute card: value_type + informed-scores header, the
// detection guidelines, and a values table (value · label · description ·
// examples). Ordinal attributes are indexed so the `>=` ordering is visible.
const OP_LABEL: Record<string, string> = {
  "==": "=",
  ">=": "≥",
  "<=": "≤",
  in: "in",
  present: "is present",
};

function fmtValue(v: any): string {
  if (v === undefined || v === null) return "";
  if (Array.isArray(v)) return v.map((x) => String(x)).join(", ");
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function AttributeBody({
  sd,
  category,
  attr,
}: {
  sd: any;
  category: Category;
  attr: string;
}) {
  // The attribute value whose labelled segments are currently expanded.
  const [active, setActive] = useState<string | null>(null);
  if (!sd || typeof sd !== "object") return null;

  if (sd.kind === "term_levels" && sd.levels && typeof sd.levels === "object") {
    const levels = Object.entries(sd.levels as Record<string, unknown>);
    return (
      <div className="attr-body">
        <div className="muted small">term levels — score band → terms</div>
        <div className="attr-table-wrap">
          <table className="attr-table">
            <thead>
              <tr>
                <th>level</th>
                <th>terms</th>
              </tr>
            </thead>
            <tbody>
              {levels.map(([lvl, terms]) => {
                const n = Number(lvl);
                return (
                  <tr key={lvl}>
                    <td>
                      <ScoreBadge score={Number.isFinite(n) ? n : 0} />
                    </td>
                    <td className="mono small">
                      {(Array.isArray(terms) ? terms : [])
                        .map((t) => String(t))
                        .join(", ")}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  // attribute_def (default shape)
  const values: any[] = Array.isArray(sd.values) ? sd.values : [];
  const scores: any[] = Array.isArray(sd.scores_informed) ? sd.scores_informed : [];
  const ordinal = sd.value_type === "ordinal";
  return (
    <div className="attr-body">
      <div className="row" style={{ gap: 8 }}>
        <Badge tone="gray">{String(sd.value_type ?? "—")}</Badge>
        {scores.length > 0 && (
          <span className="row" style={{ gap: 4 }}>
            <span className="muted small">informs</span>
            {scores.map((s, i) => (
              <ScoreBadge key={i} score={Number(s)} />
            ))}
          </span>
        )}
      </div>
      {sd.guidelines && <p className="attr-guide small">{String(sd.guidelines)}</p>}
      {values.length > 0 && (
        <div className="attr-table-wrap">
          <table className="attr-table">
            <thead>
              <tr>
                {ordinal && <th>#</th>}
                <th>value</th>
                <th>label</th>
                <th>description</th>
                <th>rules</th>
                <th>examples</th>
              </tr>
            </thead>
            <tbody>
              {values.map((v, i) => {
                const obj = v && typeof v === "object" ? v : { value: v };
                const ex = Array.isArray(obj.examples) ? obj.examples : [];
                const rules = Array.isArray(obj.rules) ? obj.rules : [];
                const valStr = String(obj.value ?? "");
                return (
                  <tr key={i}>
                    {ordinal && <td className="muted mono">{i}</td>}
                    <td className="mono">
                      {/* Click a value to see the segments labelled with it. */}
                      <button
                        onClick={() =>
                          setActive((cur) => (cur === valStr ? null : valStr))
                        }
                        title="Show segments labelled with this value"
                        style={{
                          background: "none",
                          border: "none",
                          padding: 0,
                          cursor: "pointer",
                          font: "inherit",
                          color: "inherit",
                          textDecoration: "underline dotted",
                        }}
                      >
                        {valStr}
                      </button>
                    </td>
                    <td>{String(obj.label ?? "")}</td>
                    <td className="small">{String(obj.description ?? "")}</td>
                    <td className="small">
                      {rules.length > 0 ? (
                        <ul style={{ margin: 0, paddingLeft: 16, display: "grid", gap: 3, maxWidth: 360 }}>
                          {rules.map((r: any, j: number) => (
                            <li key={j}>{String(r)}</li>
                          ))}
                        </ul>
                      ) : (
                        <span className="muted">—</span>
                      )}
                    </td>
                    <td className="small muted">
                      {ex.map((e: any) => String(e)).join("; ")}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {active != null && (
        <SegmentTrackPanel
          key={`${category}-${attr}-${active}`}
          title={`Segments where ${attr} = ${active}`}
          fetcher={() => getAttributeValueSegments(category, attr, active)}
          onClose={() => setActive(null)}
        />
      )}
    </div>
  );
}

// ── Decision rule tree diagram (hand-drawn SVG) ───────────────────────────
// A priority cascade: each rule node branches match → a score leaf, else →
// the next rule below, terminating in the default score leaf.
const DT = {
  MARGIN: 24,
  RULE_W: 320,
  LEAF_W: 190,
  COL_GAP: 100,
  V_GAP: 34,
  LEAF_H: 66,
};

function ruleHeight(rule: any): number {
  const n = Array.isArray(rule?.when) ? rule.when.length : 0;
  return Math.max(64, 34 + Math.max(1, n) * 22 + 12);
}

const vPath = (x1: number, y1: number, x2: number, y2: number) => {
  const my = (y1 + y2) / 2;
  return `M ${x1} ${y1} C ${x1} ${my}, ${x2} ${my}, ${x2} ${y2}`;
};
const hPath = (x1: number, y1: number, x2: number, y2: number) => {
  const mx = (x1 + x2) / 2;
  return `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
};

function Pill({
  x,
  y,
  text,
  kind,
}: {
  x: number;
  y: number;
  text: string;
  kind: "match" | "else";
}) {
  const w = text.length * 6.4 + 16;
  return (
    <g className={`dt-pill ${kind}`}>
      <rect x={x - w / 2} y={y - 9} width={w} height={18} rx={9} />
      <text x={x} y={y + 4} textAnchor="middle">
        {text}
      </text>
    </g>
  );
}

function LeafNode({
  x,
  y,
  score,
  note,
  label,
}: {
  x: number;
  y: number;
  score: number;
  note?: string;
  label: string;
}) {
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={DT.LEAF_W}
        height={DT.LEAF_H}
        rx={8}
        fill={scoreColor(score)}
      />
      <foreignObject x={x} y={y} width={DT.LEAF_W} height={DT.LEAF_H}>
        <div className="dt-leafbox">
          <div className="dt-leaf-head">
            <span className="dt-leaf-score">{score}</span>
            <span className="dt-leaf-tag">{label}</span>
          </div>
          {note && <div className="dt-leaf-note">{note}</div>}
        </div>
      </foreignObject>
    </g>
  );
}

function RuleNode({
  x,
  y,
  h,
  index,
  rule,
  selected,
  onClick,
}: {
  x: number;
  y: number;
  h: number;
  index: number;
  rule: any;
  selected: boolean;
  onClick: () => void;
}) {
  const conds: any[] = Array.isArray(rule?.when) ? rule.when : [];
  return (
    <g className={`dt-node${selected ? " selected" : ""}`} onClick={onClick}>
      <rect
        className="dt-node-rect"
        x={x}
        y={y}
        width={DT.RULE_W}
        height={h}
        rx={10}
      />
      <foreignObject x={x} y={y} width={DT.RULE_W} height={h}>
        <div className="dt-nodebox">
          <div className="dt-rulehead">
            <span>Rule #{index + 1}</span>
            <span className="dt-rulescore">→ score {String(rule?.score ?? "—")}</span>
          </div>
          {conds.length === 0 ? (
            <div className="dt-cond muted">always</div>
          ) : (
            conds.map((c, i) => (
              <div className="dt-cond" key={i}>
                <span className="dt-attr">{String(c?.attribute ?? "?")}</span>{" "}
                <span className="dt-op">{OP_LABEL[c?.op] ?? String(c?.op ?? "")}</span>
                {c?.op !== "present" && (
                  <span className="dt-val"> {fmtValue(c?.value)}</span>
                )}
              </div>
            ))
          )}
        </div>
      </foreignObject>
    </g>
  );
}

function DecisionTreeDiagram({
  node,
  selected,
  onSelect,
}: {
  node: Policy | null;
  selected: number | null;
  onSelect: (i: number) => void;
}) {
  const sd = node?.structured_data;
  if (!node || !sd || sd.kind !== "decision_tree") return null;
  const rules: any[] = Array.isArray(sd.rules) ? sd.rules : [];
  const def = Number(sd.default ?? 0);

  const ruleX = DT.MARGIN;
  const leafX = ruleX + DT.RULE_W + DT.COL_GAP;

  // Vertical layout: rule nodes stack in the left column, match-leaves sit to
  // their right, and the else-chain flows straight down to the default leaf.
  let y = DT.MARGIN;
  const pos = rules.map((r) => {
    const h = ruleHeight(r);
    const p = { y, h };
    y += Math.max(h, DT.LEAF_H) + DT.V_GAP;
    return p;
  });
  const defaultY = y;
  const width = leafX + DT.LEAF_W + DT.MARGIN;
  const height = defaultY + DT.LEAF_H + DT.MARGIN;

  const elseX = ruleX + DT.RULE_W / 2; // else edges leave the node's bottom-centre

  return (
    <div style={{ marginTop: 16 }}>
      <div className="row spread">
        <h3 style={{ margin: 0 }}>Decision rule tree</h3>
        <span className="muted small">
          <span className="mono">{node.policy_id}</span> v{node.version} · priority
          cascade — first match wins
        </span>
      </div>
      <div className="dt-wrap">
        {rules.length === 0 ? (
          <div className="state">No rules defined — always scores {def}.</div>
        ) : (
          <svg
            className="dt-svg"
            width={width}
            height={height}
            viewBox={`0 0 ${width} ${height}`}
          >
            {/* edges first, then pills, then nodes */}
            {pos.map((p, i) => {
              const cy = p.y + p.h / 2;
              const nextY =
                i < rules.length - 1 ? pos[i + 1].y : defaultY;
              const nextX =
                i < rules.length - 1 ? elseX : ruleX + DT.LEAF_W / 2;
              return (
                <g key={`e${i}`}>
                  <path
                    className="dt-edge match"
                    d={hPath(
                      ruleX + DT.RULE_W,
                      cy,
                      leafX,
                      p.y + DT.LEAF_H / 2,
                    )}
                  />
                  <path
                    className="dt-edge"
                    d={vPath(elseX, p.y + p.h, nextX, nextY)}
                  />
                </g>
              );
            })}
            {pos.map((p, i) => {
              const nextY = i < rules.length - 1 ? pos[i + 1].y : defaultY;
              const cy = p.y + p.h / 2;
              return (
                <g key={`p${i}`}>
                  <Pill
                    x={(ruleX + DT.RULE_W + leafX) / 2}
                    y={(cy + p.y + DT.LEAF_H / 2) / 2}
                    text="match"
                    kind="match"
                  />
                  <Pill
                    x={elseX}
                    y={(p.y + p.h + nextY) / 2}
                    text="else"
                    kind="else"
                  />
                </g>
              );
            })}
            {pos.map((p, i) => (
              <LeafNode
                key={`l${i}`}
                x={leafX}
                y={p.y}
                score={Number(rules[i]?.score ?? 0)}
                note={rules[i]?.note ? String(rules[i].note) : undefined}
                label="match"
              />
            ))}
            <LeafNode
              x={ruleX}
              y={defaultY}
              score={def}
              label="default"
              note="no rule matched"
            />
            {pos.map((p, i) => (
              <RuleNode
                key={`n${i}`}
                x={ruleX}
                y={p.y}
                h={p.h}
                index={i}
                rule={rules[i]}
                selected={selected === i}
                onClick={() => onSelect(i)}
              />
            ))}
          </svg>
        )}
      </div>
      <div className="muted small" style={{ marginTop: 6 }}>
        Click a rule node to highlight the attribute cards its conditions read.
      </div>
    </div>
  );
}

function PolicyNode({
  node,
  depth,
  highlighted,
  category,
}: {
  node: Policy;
  depth: number;
  highlighted: Set<string>;
  category: Category;
}) {
  const isAttr = node.type === "attribute";
  const key = attrKey(node);
  const hot = isAttr && highlighted.has(key);
  return (
    <div style={{ marginLeft: depth * 20 }}>
      <div className={`card${hot ? " highlight" : ""}`}>
        <div className="row spread">
          <div className="row">
            <Badge tone={TYPE_TONE[node.type] ?? "gray"}>{node.type}</Badge>
            {isAttr ? (
              <strong className="mono">{key}</strong>
            ) : (
              <span className="mono small">{node.policy_id}</span>
            )}
            <span className="muted small">v{node.version}</span>
          </div>
          <Badge tone={node.status === "active" ? "ok" : "gray"}>{node.status}</Badge>
        </div>
        <p style={{ margin: "8px 0 0" }}>{node.text}</p>
        {isAttr && (
          <AttributeBody sd={node.structured_data} category={category} attr={key} />
        )}
        {node.structured_ref && (
          <div className="muted small mono">structured_ref: {node.structured_ref}</div>
        )}
      </div>
      {node.children?.map((c) => (
        <div key={c.policy_id} style={{ marginTop: 8 }}>
          <PolicyNode
            node={c}
            depth={depth + 1}
            highlighted={highlighted}
            category={category}
          />
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
      {req.target_policy_id && (
        <div className="small muted" style={{ marginTop: 4 }}>
          Targets: <span className="mono">{req.target_policy_id}</span>
        </div>
      )}
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
