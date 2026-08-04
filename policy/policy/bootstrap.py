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
    "the labelling policy: the attribute schema a labeller needs and an "
    "attribute-based decision tree mapping attribute values to the 0..5 score. "
    "Ground every attribute and rule in the rubric's bands. Respond with a "
    "single JSON object only, no prose."
)

# Concrete example: anchors the output shape/quality and cuts the model's
# reasoning (flash at temp 0 otherwise emits degenerate or truncated JSON).
_ATTR_EXAMPLE = (
    '{"attributes":[{"name":"casino_game_present","value_type":"boolean",'
    '"values":null,"guidelines":"A recognisable casino game (slots, roulette, '
    'poker, blackjack) is shown being played.","scores_informed":[2,3,4]},'
    '{"name":"stake_type","value_type":"categorical","values":["none",'
    '"in_game_currency","real_money"],"guidelines":"What is wagered: nothing, '
    'in-game currency, or real money.","scores_informed":[2,4,5]}]}'
)
_RULE_EXAMPLE = (
    '{"when":[{"attribute":"stake_type","op":"==","value":"real_money"}],'
    '"score":5,"note":"Functional real-money gambling."}'
)


def _judge_json(orch, prompt: str) -> dict:
    """One orchestrator JSON call with a single retry — flash occasionally emits
    an empty/truncated reply, and a retry usually recovers a complete one."""
    for _ in range(2):
        parsed = _parse_json(orch.judge(_SYNTH_SYS, prompt))
        if parsed:
            return parsed
    return {}


def _synth_attributes(orch, cat: str, rubric: str, obs: str) -> list[dict]:
    """Ask the orchestrator to define the attributes a labeller needs."""
    prompt = (
        f"Category: {cat}\n\n"
        f"Scoring rubric (0..5 age bands):\n{rubric}\n\n"
        f"Observed shots:\n{obs}\n\n"
        "Define 4-6 attributes a labeller must observe to decide the 0..5 score "
        "for THIS category. Each attribute is an OBSERVABLE property of a shot, "
        "never the score itself. Match this exact shape:\n" + _ATTR_EXAMPLE + "\n"
        "value_type is boolean/categorical/level/count; values lists the allowed "
        "values for categorical/level (else null); scores_informed are the rubric "
        "bands the attribute is evidence for. One-sentence guidelines. Output the "
        "JSON only."
    )
    raw = _judge_json(orch, prompt).get("attributes") or []
    if isinstance(raw, dict):  # some replies key attributes by name
        raw = [{"name": k, **v} for k, v in raw.items() if isinstance(v, dict)]
    return [a for a in raw if isinstance(a, dict) and a.get("name")]


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
        "Use only the attribute names/values above; short note; JSON only."
    )
    d = _judge_json(orch, prompt)
    if "when" in d and "score" in d:
        d["score"] = max(0, min(5, int(score)))  # trust the requested band
        return d
    return None


def _synth_decision_tree(orch, cat: str, rubric: str, attrs: list[dict]) -> dict:
    """Priority-ordered decision tree over the defined attributes, built one rule
    per rubric band (highest score first) so each orchestrator reply is small
    enough to complete. Band 0 (no content) is the default."""
    schema_json = json.dumps([
        {"name": a.get("name"),
         "value_type": a.get("value_type") or a.get("type"),
         "values": a.get("values") or a.get("enum")}
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
