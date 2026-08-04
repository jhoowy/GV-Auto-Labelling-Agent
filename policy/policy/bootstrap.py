"""Bootstrap — establish the initial policy set.

Reuses the normal mechanism (labelling agent + propose_policy_change queue),
run intensively over bootstrap videos, then converged.

    seed(PEGI -> rubric nodes = v0) -> label bootstrap videos -> cluster
    proposals into edge-case candidates -> human review -> iterate until
    cross-sample label variance converges -> policy-set v1
"""
from __future__ import annotations

import json
import logging
import re

from schemas import Category, Policy
from schemas.enums import PolicyType
from tools import policy_store

log = logging.getLogger(__name__)


# PEGI-aligned 0..5 score bands (score: age tier). One SCORING node per
# category = one RAG chunk holding the full 0..5 rubric. Human-readable text
# is what the agent retrieves and cites; structured data (word lists, etc.)
# lives outside this text under a later ATTRIBUTE node.
_PEGI_RUBRICS: dict[Category, str] = {
    Category.GAMBLING: (
        "Gambling scoring rubric (age score 0..5).\n"
        "0 (3+): No gambling content of any kind — no betting, casino games, "
        "wagering imagery, or chance-based paid rewards.\n"
        "1 (7+): Incidental chance-based play with no stakes (board-game dice, "
        "spin wheels) framed as child-friendly.\n"
        "2 (12+): Simulated gambling — casino-style games (cards, roulette, "
        "slots) played with in-game currency only, not teaching real betting.\n"
        "3 (16+): Frequent simulated gambling or loot boxes / random paid "
        "rewards presented as a core, rewarded activity, but no real money.\n"
        "4 (18+): Content that teaches or encourages gambling with possible "
        "real-money stakes, or realistic casino simulation mirroring real "
        "betting.\n"
        "5 (blocked): Functional real-money gambling or direct links to real "
        "betting/casino services — disallowed for distribution."
    ),
    Category.BAD_LANGUAGE: (
        "Bad language scoring rubric (age score 0..5).\n"
        "0 (3+): No profanity; language suitable for young children.\n"
        "1 (7+): Very mild expletives only (e.g. 'damn', 'hell'), rare and "
        "non-aggressive.\n"
        "2 (12+): Mild swearing without sexual expletives; occasional moderate "
        "profanity.\n"
        "3 (16+): Strong language more than occasionally, including aggressive "
        "insults, but excluding sexual expletives.\n"
        "4 (18+): Strong/crude language including sexual expletives or slurs "
        "used frequently.\n"
        "5 (blocked): Extreme hate speech or slurs promoting violence or "
        "discrimination against protected groups — disallowed."
    ),
    Category.SEX: (
        "Sex scoring rubric (age score 0..5).\n"
        "0 (3+): No sexual content — no nudity, innuendo, or suggestive "
        "posing.\n"
        "1 (7+): Very mild romantic content (hand-holding, brief kissing) with "
        "no suggestion.\n"
        "2 (12+): Sexual innuendo, suggestive posturing, or non-detailed "
        "references to sex; brief non-sexual nudity.\n"
        "3 (16+): Erotic nudity or implied sexual activity without explicit "
        "detail.\n"
        "4 (18+): Explicit depiction of sexual activity or strong "
        "erotic/pornographic imagery.\n"
        "5 (blocked): Sexual content involving minors or non-consensual acts — "
        "disallowed."
    ),
}


def seed_from_pegi(categories: list[Category] | None = None) -> None:
    """Generate v0 scoring-rubric nodes from PEGI criteria.

    Persists one SCORING Policy per category via upsert_policy (deterministic
    id `<category>.scoring`, so re-seeding edits in place rather than
    duplicating). These nodes are the policy-set v0 baseline. No ATTRIBUTE
    nodes are seeded — structured attributes (e.g. a profanity term list by
    level) are drafted by the agent during bootstrap, not pre-seeded.
    """
    categories = categories or list(_PEGI_RUBRICS.keys())
    for cat in categories:
        text = _PEGI_RUBRICS.get(cat)
        if text is None:
            log.warning("no PEGI rubric defined for category %s; skipping", cat)
            continue
        policy_store.upsert_policy(Policy(
            policy_id=f"{cat.value}.scoring",
            type=PolicyType.SCORING,
            category=cat,
            text=text,
        ))
        log.info("seeded PEGI scoring rubric for %s", cat.value)


def _cluster_proposals(reqs: list) -> list[list]:
    """Group queued change requests into edge-case candidates.

    Naive token-overlap (Jaccard) clustering — no model needed and robust when
    the embedding server is down. Each cluster is one candidate EDGE_CASE rule
    a human reviews before it enters the tree.
    """
    threshold = 0.4
    clusters: list[list] = []
    tokenised = [(r, set(r.proposed_change.lower().split())) for r in reqs]
    for r, toks in tokenised:
        placed = False
        for cluster in clusters:
            _, ctoks = cluster[0]
            union = toks | ctoks
            jac = len(toks & ctoks) / len(union) if union else 0.0
            if jac >= threshold:
                cluster.append((r, toks))
                placed = True
                break
        if not placed:
            clusters.append([(r, toks)])
    return [[r for r, _ in cluster] for cluster in clusters]


def _parse_json(text: str) -> dict:
    """Lenient parse — the orchestrator may wrap JSON in prose/fences."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                return {}
    return {}


def _scoring_text(category: str) -> str:
    """Human-readable scoring rubric for a category, from the live tree."""
    for node in policy_store.get_policy_tree(category):
        if node.type == PolicyType.SCORING:
            return node.text
    return ""


# Bound the digest so the orchestrator's thinking + JSON output fit its output
# token cap; very long videos are sampled evenly rather than dropped from the end.
_MAX_SHOTS = 24
_SUMMARY_CAP = 240
_TRANSCRIPT_CAP = 200


def _observations(segments: list) -> str:
    """One line per shot: idx, summary, ASR, and base (ingestion) attributes.

    Evenly sub-samples to _MAX_SHOTS and truncates long fields to keep the prompt
    small — a large prompt inflates the orchestrator's hidden reasoning and can
    starve the JSON reply of output tokens."""
    if len(segments) > _MAX_SHOTS:
        step = len(segments) / _MAX_SHOTS
        segments = [segments[int(i * step)] for i in range(_MAX_SHOTS)]
    lines: list[str] = []
    for seg in segments:
        attrs = ", ".join(
            f"{a.key}={a.value}" for a in getattr(seg, "base_attributes", [])
        )
        parts = [f"shot {seg.idx}"]
        if seg.summary:
            parts.append(f"summary: {seg.summary.strip()[:_SUMMARY_CAP]}")
        if seg.transcript:
            parts.append(f"transcript: {seg.transcript.strip()[:_TRANSCRIPT_CAP]}")
        if attrs:
            parts.append(f"base_attributes: {attrs[:200]}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


_SYNTH_SYS = (
    "You are a content-moderation policy engineer. From observed gameplay-video "
    "shots and a PEGI-aligned 0..5 scoring rubric for ONE category, you design "
    "the labelling policy: a small set of GENERAL, observable attributes "
    "(signals) a labeller can tag in one pass, and an attribute-based decision "
    "tree mapping their values to the 0..5 score. Ground every attribute and "
    "rule in the rubric's bands. Respond with a single JSON object only, no "
    "prose."
)

# What separates a reusable signal from a baked-in verdict. Kept terse: flash
# spends most of its output budget on hidden reasoning, and a long principle
# inflates that reasoning until the JSON reply truncates.
_ATTR_PRINCIPLE = (
    "An ATTRIBUTE is a GENERAL, reusable signal — a closed enum (categorical, or "
    "ordinal low..high) a labeller can point at in ONE frame — NEVER a verdict. "
    "Put specificity in the ENUM VALUES of one general attribute, not in many "
    "narrow booleans (NOT is_slot_machine; instead one `gambling_activity` with "
    "values none/simulated_casino/loot_box/real_money_casino). Ban names "
    "containing is_/core/frequent/functional/real/enough/primary — those are "
    "judgments; express them as an ordinal value or a decision-tree rule."
)

# Attribute synthesis is split so every flash reply stays tiny (a rich all-in-one
# reply truncates): stage 1 draws only names + value_type + scores; stage 2 fills
# each attribute's enum/guidelines one attribute at a time.
_ATTR_SKELETON_EXAMPLE = (
    '{"attributes":['
    '{"name":"gambling_activity","value_type":"categorical","scores_informed":'
    '[1,2,3,4,5]},'
    '{"name":"stake_severity","value_type":"ordinal","scores_informed":[2,4,5]},'
    '{"name":"cashout_available","value_type":"boolean","scores_informed":[4,5]}'
    ']}'
)
_DETAIL_EXAMPLE = (
    '{"guidelines":"Which gambling-like activity, if any, is shown.","values":['
    '{"value":"none","label":"None","description":"No gambling or chance-based '
    'reward mechanic on screen.","examples":["a platformer level"]},'
    '{"value":"simulated_casino","label":"Simulated casino","description":"A '
    'casino game played with in-game currency.","examples":["poker with chips"]}'
    ']}'
)
_RULE_EXAMPLE = (
    '{"when":[{"attribute":"stake_severity","op":">=","value":"real"}],'
    '"score":5,"note":"Real-money stakes reach the highest band."}'
)


def _judge_json(orch, prompt: str) -> dict:
    """One orchestrator JSON call with a single retry — flash occasionally emits
    an empty/truncated reply, and a retry usually recovers a complete one."""
    for _ in range(2):
        parsed = _parse_json(orch.judge(_SYNTH_SYS, prompt))
        if parsed:
            return parsed
    return {}


def _salvage_objects(text: str) -> list[dict]:
    """Recover every balanced `{...}` JSON object from a possibly-truncated reply.

    flash spends most of its output budget on hidden reasoning, so a list reply
    can be cut off mid-array; a stack scan still yields the complete element
    objects (the unterminated outer object / last element are simply skipped)."""
    objs: list[dict] = []
    stack: list[int] = []
    for i, ch in enumerate(text or ""):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            try:
                obj = json.loads(text[start:i + 1])
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, dict):
                objs.append(obj)
    return objs


def _synth_attributes(orch, cat: str, rubric: str, obs: str) -> list[dict]:
    """Define the GENERAL observable attributes a labeller needs.

    Two tiny reply stages keep flash inside its output budget (one rich all-in-one
    reply truncates under its hidden reasoning): (1) a skeleton naming 4-6 general
    attributes with only value_type + scores; (2) one call per attribute that
    fills its enum values (with per-value copy) and detection guidelines."""
    prompt = (
        f"Category: {cat}\n\n"
        f"Scoring rubric (0..5 age bands):\n{rubric}\n\n"
        f"Observed shots:\n{obs}\n\n"
        f"{_ATTR_PRINCIPLE}\n\n"
        "Name 4-6 GENERAL observable attributes a labeller tags to decide the "
        "0..5 score for THIS category. Give ONLY name, value_type "
        "(boolean/categorical/ordinal) and scores_informed (the rubric bands the "
        "attribute is evidence for) — the enum values come later. Match this "
        "exact shape:\n" + _ATTR_SKELETON_EXAMPLE + "\n"
        "Keep it short; output the JSON only."
    )
    raw_text, parsed = "", {}
    for _ in range(2):
        raw_text = orch.judge(_SYNTH_SYS, prompt)
        parsed = _parse_json(raw_text)
        if parsed.get("attributes"):
            break
    raw = parsed.get("attributes") or []
    if isinstance(raw, dict):  # some replies key attributes by name
        raw = [{"name": k, **v} for k, v in raw.items() if isinstance(v, dict)]
    attrs = [a for a in raw if isinstance(a, dict) and a.get("name")]
    if not attrs:  # truncated reply -> salvage the complete attribute objects
        attrs = [o for o in _salvage_objects(raw_text)
                 if o.get("name") and o.get("value_type")]
    for a in attrs:
        _synth_detail(orch, cat, rubric, a)
    return attrs


def _synth_detail(orch, cat: str, rubric: str, attr: dict) -> None:
    """Fill one attribute's `guidelines` and (for categorical/ordinal) its closed
    enum `values` — a small per-attribute reply. Mutates `attr` in place; boolean
    attributes keep `values=None`. Falls back to salvaged value objects when the
    reply truncates, and to a bare guideline if none arrives."""
    name = attr.get("name")
    vtype = attr.get("value_type") or attr.get("type") or "categorical"
    is_enum = vtype in ("categorical", "ordinal")
    prompt = (
        f"Category: {cat}. Attribute '{name}' (value_type {vtype}).\n"
        f"Scoring rubric:\n{rubric}\n\n"
        + ("Give its GENERAL detection `guidelines` (one line) and a CLOSED "
           "`values` enum" + (" in ASCENDING order low..high" if vtype == "ordinal"
                              else "") + ", each value with a one-line description "
           "and 0-2 short frame examples. Keep specificity in these values, not in "
           "the name. Output ONLY this JSON:\n" + _DETAIL_EXAMPLE
           if is_enum else
           "Give its GENERAL detection `guidelines` (one line). Output ONLY "
           '{"guidelines":"..."}.')
    )
    raw_text = orch.judge(_SYNTH_SYS, prompt)
    parsed = _parse_json(raw_text)
    attr["guidelines"] = (parsed.get("guidelines")
                          or attr.get("guidelines") or f"Observe {name}.")
    if not is_enum:
        attr["values"] = None
        return
    got = parsed.get("values")
    if not got:  # truncated -> salvage the complete value objects
        got = [o for o in _salvage_objects(raw_text) if "value" in o]
    seen, values = set(), []
    for v in got:
        if not isinstance(v, dict):
            continue
        key = str(v.get("value") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        values.append({
            "value": key, "label": v.get("label") or key,
            "description": v.get("description") or "",
            "examples": list(v.get("examples") or []),
        })
    attr["values"] = values or None


def _rubric_bands(rubric: str) -> dict[int, str]:
    """Parse the rubric text into {score -> band description}."""
    bands: dict[int, str] = {}
    for ln in rubric.splitlines():
        m = re.match(r"\s*([0-5])\b[^:]*:\s*(.+)", ln)
        if m:
            bands[int(m.group(1))] = m.group(2).strip()
    return bands


def _synth_rule(orch, cat: str, schema_json: str, score: int, desc: str) -> dict | None:
    """One decision rule for one score band. Kept to a single tiny reply per call
    because the orchestrator spends most of its token budget on hidden reasoning,
    so a whole tree in one reply truncates; one rule always fits."""
    prompt = (
        f"Category: {cat}. Attributes:\n{schema_json}\n\n"
        f"Score {score} means: {desc}\n\n"
        f"Give ONE decision rule whose conditions over the attributes above "
        f"select score {score}. Output ONLY this JSON object:\n" + _RULE_EXAMPLE + "\n"
        "Each condition is {attribute,op,value}; op is ==, in, present, or (for "
        "ORDINAL attributes only, comparing against a listed value) >= / <=. Use "
        "only the attribute names/values above; short note; JSON only."
    )
    d = _judge_json(orch, prompt)
    if "when" in d and "score" in d:
        d["score"] = max(0, min(5, int(score)))  # trust the requested band
        return d
    return None


def _value_keys(values) -> list | None:
    """The bare enum keys (`value` field) for the rule prompt — the model needs
    the allowed values, not the full per-value copy. Accepts the rich dict form
    or a plain list of strings."""
    if not values:
        return None
    return [v.get("value") if isinstance(v, dict) else v for v in values]


def _synth_decision_tree(orch, cat: str, rubric: str, attrs: list[dict]) -> dict:
    """Priority-ordered decision tree over the defined attributes, built one rule
    per rubric band (highest score first) so each orchestrator reply is small
    enough to complete. Band 0 (no content) is the default."""
    schema_json = json.dumps([
        {"name": a.get("name"),
         "value_type": a.get("value_type") or a.get("type"),
         "values": _value_keys(a.get("values") or a.get("enum"))}
        for a in attrs
    ])
    bands = _rubric_bands(rubric) or {s: "" for s in range(6)}
    rules: list[dict] = []
    for score in sorted((s for s in bands if s > 0), reverse=True):
        rule = _synth_rule(orch, cat, schema_json, score, bands[score])
        if rule:
            rules.append(rule)
    return {"default": 0, "rules": rules}


def synthesize_category_policy(
    category: str, segments: list, *, orchestrator=None
) -> dict:
    """Draft a category's attribute schema + decision-rule tree from data.

    Prompts the Gemini orchestrator with the explored shots' observations and the
    category's scoring rubric in grounded steps: (a) one call defines the
    attributes a labeller needs to decide the 0..5 score (name, value_type,
    allowed values, detection guidelines, the score bands each informs), then (b)
    one call per rubric band builds a priority-ordered decision tree over exactly
    those attributes. Attribute defs are upserted as ATTRIBUTE nodes and the tree
    as the category's DECISION_RULE node — direct DRAFT writes (the whole tree is
    human-reviewed before a policy-set v1 snapshot). Keeping every reply small is
    deliberate: the orchestrator spends most of its output-token budget on hidden
    reasoning, so a large single reply truncates. Idempotent: re-running refines
    in place (upsert bumps the node version)."""
    cat = getattr(category, "value", category)
    orch = orchestrator or _get_agent_llm()
    rubric = _scoring_text(cat)
    obs = _observations(segments)

    attrs = _synth_attributes(orch, cat, rubric, obs)
    names: list[str] = []
    for a in attrs:
        name = a["name"]
        policy_store.upsert_attribute_definition(
            cat, name,
            value_type=a.get("value_type") or a.get("type") or "categorical",
            guidelines=a.get("guidelines") or a.get("description") or "",
            scores_informed=a.get("scores_informed") or [],
            values=a.get("values") or a.get("enum"),
            examples=a.get("examples"),
        )
        names.append(name)

    tree = _synth_decision_tree(orch, cat, rubric, attrs)
    rules = tree.get("rules") or []
    default = int(tree.get("default", 0) or 0)
    policy_store.upsert_decision_rule(cat, rules, default)

    log.info(
        "bootstrap: synthesised %d attribute(s) + %d rule(s) for %s",
        len(names), len(rules), cat,
    )
    return {
        "category": cat,
        "attributes": names,
        "n_rules": len(rules),
        "default": default,
    }


def _get_agent_llm():
    from models import get_agent_llm
    return get_agent_llm()


def run_bootstrap(video_ids: list[str]) -> dict:
    """Drive the labelling loop over bootstrap data to draft a structured policy
    set. Returns a summary of the draft (seed rubrics + clustered candidates).

    Bootstrap has no pre-labelled data, so cross-data precedent retrieval is
    disabled (`label_video(..., bootstrap=True)`); the agent instead proposes
    policy gaps that queue for human review.

    Flow:
      1. SEED: ensure PEGI v0 scoring rubrics exist.
      2. RUN: label each bootstrap video with precedent retrieval OFF; JUDGE
         proposes ATTRIBUTE / EDGE_CASE gaps, SIDE_FX queues them.
      3. SYNTHESISE: per active category, draft an attribute schema + decision
         tree from the explored shots (DRAFT nodes, not snapshotted).
      4. CLUSTER: group queued proposals into candidate rules.
      5. REVIEW (external, human-gated): approve/reject via resolve_change_request;
         approved candidates become nodes and the tree is snapshotted as v1.
         Nothing here auto-applies changes or auto-snapshots.
    """
    from labelling import label_video
    from models import base_config
    from tools import storage

    seed_from_pegi()

    for vid in video_ids:
        log.info("bootstrap: labelling %s (retrieval disabled)", vid)
        label_video(vid, bootstrap=True)

    # Synthesise a per-category attribute schema + decision tree from the shots
    # explored above (draft nodes, human-reviewed before a policy-set v1 snapshot).
    segments = [seg for vid in video_ids for seg in storage.get_segments(vid)]
    categories = base_config().get("policy", {}).get("categories", [])
    synthesised = {}
    for cat in categories:
        try:
            synthesised[cat] = synthesize_category_policy(cat, segments)
        except Exception as e:  # noqa: BLE001 - one category must not abort the run
            log.warning("bootstrap: synthesis failed for %s (%s)", cat, e)

    proposals = policy_store.list_change_requests(status="queued")
    clusters = _cluster_proposals(proposals)
    log.info(
        "bootstrap: %d queued proposals grouped into %d candidate rule(s); "
        "awaiting human review before policy-set v1",
        len(proposals), len(clusters),
    )
    return {
        "videos": len(video_ids),
        "seeded_categories": [c.value for c in _PEGI_RUBRICS],
        "synthesised": synthesised,
        "n_proposals": len(proposals),
        "n_candidates": len(clusters),
        "candidates": [
            {"size": len(c), "example": c[0].proposed_change if c else ""}
            for c in clusters
        ],
    }
