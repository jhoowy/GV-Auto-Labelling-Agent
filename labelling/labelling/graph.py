"""LangGraph orchestrator — structured state machine with fixed stages, each
with a restricted tool set. The window slides until all segments are labelled.

    LOAD -> RETRIEVE -> JUDGE -> SIDE_FX -> COMMIT
                          |
             (loop to next window until done)

  RETRIEVE  policies + precedents
  JUDGE     categories with attribute defs + a decision tree run an explicit
            SELECT -> EXTRACT -> DECIDE(+trajectory) -> REVIEW -> STORE pipeline
            per shot (deterministic score from the tree; REVIEW cannot override
            it); in BOOTSTRAP (no tree yet) categories with attribute defs
            EXTRACT every attribute + assign a FREE holistic score (the CART
            training set); categories with no defs fall back to holistic scoring
  SIDE_FX   propose_policy_change (queue) — incl. rule-change requests REVIEW
            flags / define_structured_attribute (bootstrap)
  COMMIT    emit labels + update the carry-over rolling summary

See docs/AGENT_WORKFLOW.md for the full definition.
"""
from __future__ import annotations

import json
import math
import re
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from schemas import Attribute, Label
from schemas.enums import SCORE_MAX, SCORE_MIN, AttributeLayer, Category

from . import tools as agent_tools
from .state import LabellingState

# The service layer (`tools.*`, `models`) pulls in `db` / model clients, so it is
# imported lazily inside the nodes that use it — build_graph() needs no server.

STAGES = ["LOAD", "RETRIEVE", "JUDGE", "SIDE_FX", "COMMIT"]

_FRAMES_PER_SHOT = 5

_JUDGE_SYS = (
    "You are a content-moderation labelling agent for gameplay videos. Judge the "
    "TARGET SHOT for EACH of these categories: {cats}. Score each on a 0-5 PEGI "
    "age score (0:3+, 1:7+, 2:12+, 3:16+, 4:18+, 5:blocked). Ground every "
    "judgement in the supplied policy nodes and cite the exact policy ids you "
    "relied on. Stay consistent with the precedent labels of similar shots unless "
    "the evidence clearly differs. You are given up to 5 sampled frames plus the "
    "shot summary and ASR text; if the frames are insufficient to judge, set "
    "need_more_frames=true. Return ONLY JSON of the form "
    '{{"judgements":[{{"category":str,"score":int,"rationale":str,'
    '"cited_policy_ids":[str],"confidence":float}}],"need_more_frames":bool}} '
    "with one judgement per category."
)

_BOOTSTRAP_SUFFIX = (
    " The policy set is still being bootstrapped from unlabelled data, so no "
    "precedents are available. Additionally report policy_gaps: content in this "
    "shot the current rubric does NOT cover well and that warrants a new "
    "ATTRIBUTE definition or EDGE_CASE rule. Add a top-level "
    '"policy_gaps":[{{"category":str,"kind":"attribute"|"edge_case",'
    '"suggestion":str,"rationale":str}}] (empty list if none).'
    " If a category would benefit from a STRUCTURED term list matched by score "
    "level (e.g. a profanity list where stronger terms map to higher scores), "
    'also add "structured_attributes":[{{"category":str,"name":str,'
    '"levels":{{"<score-level>":[terms]}},"description":str}}] (empty list if '
    "none). Levels are PEGI bands 0..5; a term at level L is evidence toward "
    "score L for that category."
)

_EXTRACT_SYS = (
    "You are a content-moderation attribute extractor for gameplay videos. For "
    "the TARGET SHOT and category '{cat}', extract a value for EACH listed "
    "attribute from the up-to-5 sampled frames plus the shot summary and ASR "
    "text. Obey each attribute's value_type and allowed values exactly (booleans "
    "as true/false). Give a short evidence string per attribute. If the sampled "
    "frames are insufficient, set need_more_frames=true. If the listed attributes "
    "and rules plainly cannot describe this shot's relevant {cat} content, set "
    "tree_fits=false and explain the gap in gap_note. Return ONLY JSON of the "
    'form {{"attributes":{{"<name>":{{"value":<value>,"evidence":str}}}},'
    '"confidence":float,"need_more_frames":bool,"tree_fits":bool,'
    '"gap_note":str}}.'
)

_SELECT_SYS = (
    "You are a content-moderation labelling agent for gameplay videos. Decide "
    "which policy CATEGORIES are relevant to the TARGET SHOT and, for each "
    "relevant category, which of its ATTRIBUTES you can label from the evidence. "
    "You are shown each category with its attribute NAMES only (not their full "
    "definitions). Select a category only if the shot plausibly contains that "
    "content; select an attribute only if the up-to-5 sampled frames + summary + "
    "ASR text let you label it. Omit categories/attributes that do not apply — "
    "anything omitted is treated as absent (empty) and left out of scoring. "
    "Return ONLY JSON of the form "
    '{"selections":[{"category":str,"attributes":[str]}]}.'
)

_REVIEW_SYS = (
    "You are a content-moderation reviewer for gameplay videos. For the TARGET "
    "SHOT you are given, per category, a score that was computed "
    "DETERMINISTICALLY from a decision tree over the extracted attributes, plus "
    "the decision trajectory (which attributes were selected, their extracted "
    "values, and which rule fired). Judge whether the score and trajectory are "
    "appropriate for the shot. You CANNOT change the score. If the score looks "
    "wrong, or the tree failed to capture the shot's relevant content, set "
    "needs_change=true and describe in change_note what the decision tree should "
    "do differently; otherwise needs_change=false. Return ONLY JSON of the form "
    '{"reviews":[{"category":str,"needs_change":bool,"change_note":str}]}.'
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _categories() -> list[str]:
    """Active category set — parameterised from policy config, not hardcoded."""
    from models import base_config

    cats = (base_config().get("policy", {}) or {}).get("categories")
    return list(cats) if cats else [c.value for c in Category]


def _clamp(score) -> int:
    try:
        s = int(score)
    except (TypeError, ValueError):
        s = SCORE_MIN
    return max(SCORE_MIN, min(SCORE_MAX, s))


def _merged_asr(seg, utterances) -> str:
    words = [u.text for u in utterances
             if u.t_end > seg.t_start and u.t_start < seg.t_end]
    return " ".join(words).strip()


def _parse_json(text: str) -> dict:
    """Lenient parse — structured output is not forced on the model."""
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


def _normalise_pins(cited: list[str], version_by_id: dict[str, int]) -> list[str]:
    """Canonicalise the model's cited policy ids into (policy_id, version) pins.

    Keeps only ids that were actually retrieved (drops hallucinated ones) and
    re-attaches the TRUE retrieved version — the model's own ",vN" suffix, if
    any, is stripped and never trusted. Order preserved, deduped.
    """
    pins: list[str] = []
    seen: set[str] = set()
    for cid in cited:
        if not cid:
            continue
        base = re.sub(r",v\d+$", "", str(cid).strip())
        ver = version_by_id.get(base)
        if ver is None:
            continue
        pin = f"{base},v{ver}"
        if pin not in seen:
            seen.add(pin)
            pins.append(pin)
    return pins


def _as_num(x):
    """Coerce to float for ordered comparison; bools/non-numerics -> None."""
    if isinstance(x, bool):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _as_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("true", "yes", "1")
    if isinstance(x, (int, float)):
        return bool(x)
    return None


def _val_eq(a, b) -> bool:
    """Type-robust equality: bool vs "true"/1, numeric vs "3", else case-fold str."""
    if isinstance(a, bool) or isinstance(b, bool):
        ba, bb = _as_bool(a), _as_bool(b)
        return ba is not None and ba == bb
    na, nb = _as_num(a), _as_num(b)
    if na is not None and nb is not None:
        return na == nb
    return str(a).strip().lower() == str(b).strip().lower()


def _ordinal_rank(keys: list, val) -> int | None:
    """Position of `val` in an ascending ordinal value list (case-insensitive)."""
    target = str(val).strip().lower()
    for i, k in enumerate(keys):
        if str(k).strip().lower() == target:
            return i
    return None


def _match_cond(cond: dict, values: dict, order: dict | None = None) -> bool:
    """One decision-tree condition against the extracted attribute values.

    `order` maps an ordinal attribute name to its ascending value keys, so `>=`
    / `<=` on a non-numeric ordinal compares by position in that order."""
    attr = cond.get("attribute")
    op = cond.get("op")
    target = cond.get("value")
    present = attr in values and values.get(attr) is not None
    if op == "present":
        return present
    if not present:
        return False
    v = values.get(attr)
    if op == "==":
        return _val_eq(v, target)
    if op in (">=", "<="):
        a, b = _as_num(v), _as_num(target)
        if a is None or b is None:  # non-numeric -> fall back to ordinal ranking
            keys = (order or {}).get(attr)
            if not keys:
                return False
            a, b = _ordinal_rank(keys, v), _ordinal_rank(keys, target)
            if a is None or b is None:
                return False
        return a >= b if op == ">=" else a <= b
    if op == "in":
        opts = target if isinstance(target, (list, tuple, set)) else [target]
        return any(_val_eq(v, o) for o in opts)
    return False


def _apply_decision_tree(rules: list, default: int, values: dict,
                         order: dict | None = None) -> tuple[int, dict | None]:
    """Evaluate a priority-ordered decision tree against extracted attribute
    values. First rule whose every `when` condition matches wins; otherwise the
    default. `order` supplies ascending value keys for ordinal attributes so
    rules can compare with `>=`. Returns (clamped score, matched rule | None)."""
    for rule in rules or []:
        conds = rule.get("when") or []
        if all(_match_cond(c, values, order) for c in conds):
            return _clamp(rule.get("score", default)), rule
    return _clamp(default), None


def _attr_value(v):
    """Coerce an extracted value into the Attribute.value union (str|float|bool)."""
    if isinstance(v, (bool, str)):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    return "unknown" if v is None else str(v)


def _segment_block(seg, asr: str, policy_attrs: list[Attribute]) -> str:
    base = ", ".join(f"{a.key}={a.value}" for a in seg.base_attributes) or "none"
    derived = ", ".join(f"{a.key}={a.value}" for a in policy_attrs) or "none"
    return (
        f"[shot {seg.idx} | {seg.segment_id} | {seg.t_start:.1f}-{seg.t_end:.1f}s]\n"
        f"summary: {seg.summary or 'n/a'}\n"
        f"asr: {asr or 'n/a'}\n"
        f"base_attributes: {base}\n"
        f"policy_attributes: {derived}"
    )


def _judge_prompt(state, seg, asr, policy_attrs, policy_text, precedents) -> str:
    seg_prec = [pr for pr in precedents if pr["segment_id"] == seg.segment_id]
    prec_text = "\n".join(
        f"- similar {pr['similar_id']}: " + ", ".join(
            f"{lbl['category']}={lbl['score']}" for lbl in pr["labels"]
        )
        for pr in seg_prec
    ) or "none"
    return (
        f"GLOBAL VIDEO OVERVIEW:\n{state.get('global_overview', '') or 'n/a'}\n\n"
        f"CARRY-OVER (confirmed so far):\n{state.get('carry_over', '') or 'none'}\n\n"
        f"TARGET SHOT (frames attached below):\n"
        f"{_segment_block(seg, asr, policy_attrs)}\n\n"
        f"RELEVANT POLICY NODES:\n{policy_text}\n\n"
        f"PRECEDENT LABELS (similar shots):\n{prec_text}\n\n"
        "Judge every category for the TARGET SHOT."
    )


def _extract_prompt(state, seg, asr, cat, attr_lines) -> str:
    return (
        f"GLOBAL VIDEO OVERVIEW:\n{state.get('global_overview', '') or 'n/a'}\n\n"
        f"CARRY-OVER (confirmed so far):\n{state.get('carry_over', '') or 'none'}\n\n"
        f"CATEGORY: {cat}\n\n"
        f"TARGET SHOT (frames attached below):\n{_segment_block(seg, asr, [])}\n\n"
        f"ATTRIBUTES TO EXTRACT:\n{attr_lines}\n\n"
        "Extract every listed attribute for the TARGET SHOT."
    )


def _select_prompt(state, seg, asr, names_by_cat: dict[str, list]) -> str:
    cats_block = "\n".join(
        f"- {cat}: {', '.join(names) or '(no attributes)'}"
        for cat, names in names_by_cat.items()
    ) or "none"
    return (
        f"GLOBAL VIDEO OVERVIEW:\n{state.get('global_overview', '') or 'n/a'}\n\n"
        f"CARRY-OVER (confirmed so far):\n{state.get('carry_over', '') or 'none'}\n\n"
        f"TARGET SHOT (frames attached below):\n{_segment_block(seg, asr, [])}\n\n"
        f"CATEGORIES AND THEIR ATTRIBUTES:\n{cats_block}\n\n"
        "Select the relevant categories and, per category, the attributes to label."
    )


def _review_prompt(state, seg, asr, items: list[dict]) -> str:
    block = "\n".join(
        f"- {it['category']}: score={it['score']}\n"
        f"    trajectory: {json.dumps(it['trajectory'], ensure_ascii=False)}"
        for it in items
    ) or "none"
    return (
        f"GLOBAL VIDEO OVERVIEW:\n{state.get('global_overview', '') or 'n/a'}\n\n"
        f"CARRY-OVER (confirmed so far):\n{state.get('carry_over', '') or 'none'}\n\n"
        f"TARGET SHOT (frames attached below):\n{_segment_block(seg, asr, [])}\n\n"
        f"DETERMINISTIC SCORES AND TRAJECTORIES:\n{block}\n\n"
        "Review each category's score and trajectory for the TARGET SHOT."
    )


def _parse_selections(parsed: dict, attr_cats: list[str],
                      names_by_cat: dict[str, list]) -> dict[str, list]:
    """Lenient parse of the SELECT call into {category: [attr_name,...]}.

    Only known categories and known attribute names survive; every attribute-
    driven category is present, defaulting to [] when the agent omits it — an
    omitted category/attribute is considered-and-empty, so the tree runs without
    those values (empty-safe -> typically the default score)."""
    sel: dict[str, list] = {c: [] for c in attr_cats}
    for item in parsed.get("selections") or []:
        if not isinstance(item, dict):
            continue
        cat = item.get("category")
        if cat not in sel:
            continue
        known = names_by_cat.get(cat, [])
        picked: list[str] = []
        for name in item.get("attributes") or []:
            if name in known and name not in picked:
                picked.append(name)
        sel[cat] = picked
    return sel


def _parse_reviews(parsed: dict, attr_cats: list[str]) -> dict[str, dict]:
    """Lenient parse of the REVIEW call into {category: {needs_change, change_note}}.
    Only known categories are kept; the review never carries a score (REVIEW
    cannot override the deterministic decision-tree score)."""
    out: dict[str, dict] = {}
    for item in parsed.get("reviews") or []:
        if not isinstance(item, dict):
            continue
        cat = item.get("category")
        if cat not in attr_cats:
            continue
        out[cat] = {"needs_change": bool(item.get("needs_change")),
                    "change_note": str(item.get("change_note") or "")}
    return out


# --------------------------------------------------------------------------- #
# stage nodes
# --------------------------------------------------------------------------- #
def _load(state: LabellingState) -> dict:
    """Populate the window; merge overlapping ASR. The head (window_stride) shots
    are committed; the rest are neighbour context. tool_trace resets per window."""
    cursor = state["cursor"]
    size = state.get("window_size", 5)
    stride = state.get("window_stride", 3)
    segs = state["all_segments"]
    utterances = state.get("utterances", [])

    window = segs[cursor:cursor + size]
    confirm = window[:stride] or window
    asr_by_segment = {s.segment_id: _merged_asr(s, utterances) for s in window}

    trace = [{"stage": "LOAD", "cursor": cursor,
              "window": [s.segment_id for s in window],
              "confirm": [s.segment_id for s in confirm]}]
    return {"window": window, "confirm": confirm,
            "asr_by_segment": asr_by_segment, "tool_trace": trace}


def _retrieve(state: LabellingState) -> dict:
    """search_policies + find_similar_segments -> retrieved_policies, precedents."""
    confirm = state.get("confirm", state.get("window", []))
    asr = state.get("asr_by_segment", {})
    query = " ".join(
        f"{s.summary or ''} {asr.get(s.segment_id, '')}" for s in confirm
    ).strip()

    policies = []
    for cat in _categories():
        policies += agent_tools.search_policies(query, cat)

    # Bootstrap has no confirmed labels yet -> precedent retrieval is disabled.
    precedents: list[dict] = []
    if not state.get("bootstrap"):
        for seg in confirm:
            for sim, labels in agent_tools.find_similar_segments(seg):
                precedents.append({
                    "segment_id": seg.segment_id,
                    "similar_id": sim.segment_id,
                    "labels": [lbl.model_dump(mode="json") for lbl in labels],
                })

    trace = state.get("tool_trace", []) + [{
        "stage": "RETRIEVE", "n_policies": len(policies),
        "n_precedents": len(precedents), "bootstrap": bool(state.get("bootstrap")),
    }]
    return {"retrieved_policies": policies, "precedents": precedents,
            "tool_trace": trace}


def _judge(state: LabellingState) -> dict:
    """Attribute-driven judging.

    Per confirm shot, the categories that have ATTRIBUTE definitions plus a
    DECISION_RULE tree run one explicit SELECT -> EXTRACT -> DECIDE -> REVIEW ->
    STORE pipeline (`_judge_pipeline`): SELECT/REVIEW are single cross-category
    calls, EXTRACT is one call per selected category, DECIDE applies the tree
    deterministically, and STORE emits one Label per category. In bootstrap no
    tree exists yet, so categories with attribute defs take the bootstrap attr
    path (`_judge_bootstrap_attrs`): extract EVERY attribute + a FREE holistic
    score, forming the CART training set. Categories with no defs (or no rules
    outside bootstrap) fall back to holistic multimodal scoring. The
    orchestrator/DB are resolved here, so build_graph() needs no server."""
    from models import get_agent_llm
    from tools import policy_store, retrieval

    orch = get_agent_llm()
    cats = _categories()
    bootstrap = bool(state.get("bootstrap"))
    confirm = state.get("confirm", [])
    asr = state.get("asr_by_segment", {})
    policies = state.get("retrieved_policies", [])
    precedents = state.get("precedents", [])

    # Per-category attribute-def nodes + decision-rule tree from the policy tree.
    cat_defs: dict[str, list] = {}
    cat_rule: dict[str, object] = {}
    for cat in cats:
        defs, rule = [], None
        for node in policy_store.get_policy_tree(cat):
            sd = node.structured_data or {}
            if sd.get("kind") == "attribute_def":
                defs.append(node)
            elif sd.get("kind") == "decision_tree" and node.policy_id == f"{cat}.rules":
                rule = node
        cat_defs[cat], cat_rule[cat] = defs, rule

    # Attribute-driven (tree) needs BOTH defs and a tree. In bootstrap no tree
    # exists yet (it is LEARNED from these labels later), so categories WITH
    # attribute defs take the bootstrap attr path — extract EVERY attribute + a
    # FREE holistic score — building the CART training set. Categories without
    # defs keep pure holistic scoring so JUDGE still works pre-synthesis.
    if bootstrap:
        attr_cats: list[str] = []
        boot_attr_cats = [c for c in cats if cat_defs[c]]
    else:
        attr_cats = [c for c in cats if cat_defs[c] and cat_rule[c]]
        boot_attr_cats = []
    fallback_cats = [c for c in cats
                     if c not in attr_cats and c not in boot_attr_cats]

    drafts: list[Label] = []
    proposals: list[dict] = []
    for seg in confirm:
        text = asr.get(seg.segment_id, "")
        # Frames are sampled ONCE per shot and reused across the pipeline's
        # SELECT / EXTRACT / REVIEW calls (and the holistic fallback below).
        frames = agent_tools.sample_frames(seg, _FRAMES_PER_SHOT)

        if attr_cats:
            ad_drafts, ad_props = _judge_pipeline(
                orch, state, seg, text, frames, attr_cats, cat_defs, cat_rule)
            drafts.extend(ad_drafts)
            proposals.extend(ad_props)

        if boot_attr_cats:
            drafts.extend(_judge_bootstrap_attrs(
                orch, state, seg, text, frames, boot_attr_cats, cat_defs,
                policies))

        if fallback_cats:
            fb_drafts, fb_props = _judge_holistic(
                orch, state, seg, text, frames, fallback_cats, policies,
                precedents, bootstrap, policy_store, retrieval)
            drafts.extend(fb_drafts)
            proposals.extend(fb_props)

    return {"draft_labels": drafts, "proposals": proposals}


def _attr_name(policy_id: str) -> str:
    """Bare attribute name from a `<category>.attr.<name>` node id."""
    return policy_id.split(".attr.", 1)[1] if ".attr." in policy_id else policy_id


def _attr_line(name: str, sd: dict, fallback_text: str) -> str:
    """One EXTRACT prompt line for an attribute: its value_type, allowed values
    (each with description AND its per-value edge-case `rules`), and detect note.
    The per-value `rules` are the case-notes shipped in the attribute def; the
    extractor must see them to pick the right value."""
    vals = sd.get("values") or []
    line = f"- {name} ({sd.get('value_type', '')})"
    if vals:
        rendered = []
        for v in vals:
            if not isinstance(v, dict):
                rendered.append(str(v))
                continue
            piece = f"{v.get('value')} — {v.get('description', '')}".rstrip(" —")
            for r in v.get("rules") or []:
                piece += f"\n      rule: {r}"
            rendered.append(piece)
        line += " — pick one of:\n    " + "\n    ".join(rendered)
    return line + f"\n    detect: {sd.get('guidelines', fallback_text)}"


def _attr_index(defs: list) -> dict:
    """Index a category's attribute-def nodes for the pipeline: name -> node, the
    ordinal ascending-value map (so rules can compare with >=), the bare name
    list (shown in SELECT), and a rendered EXTRACT line per attribute."""
    def_by_name: dict[str, object] = {}
    order: dict[str, list] = {}
    names: list[str] = []
    line_by_name: dict[str, str] = {}
    for d in defs:
        name = _attr_name(d.policy_id)
        def_by_name[name] = d
        names.append(name)
        sd = d.structured_data or {}
        vals = sd.get("values") or []
        keys = [v.get("value") if isinstance(v, dict) else v for v in vals]
        if sd.get("value_type") == "ordinal" and keys:
            order[name] = keys  # ascending -> lets rules compare with >=
        line_by_name[name] = _attr_line(name, sd, d.text)
    return {"def_by_name": def_by_name, "order": order,
            "names": names, "line_by_name": line_by_name}


def _extract_selected(orch, state, seg, asr, frames, cat, idx, names):
    """EXTRACT: one call resolving ONLY the `names` selected attributes to
    value + evidence, rendering each with its allowed values and per-value rules.
    Keeps the need_more_frames/expand_frames retry. Returns {name: {value,
    evidence}} for the selected attributes the model actually returned."""
    lines = "\n".join(idx["line_by_name"][n] for n in names)
    system = _EXTRACT_SYS.format(cat=cat)
    prompt = _extract_prompt(state, seg, asr, cat, lines)
    parsed = _parse_json(orch.judge(system, prompt, frames))
    if parsed.get("need_more_frames") and frames:
        more = agent_tools.expand_frames(seg, _FRAMES_PER_SHOT * 2)
        parsed = _parse_json(orch.judge(system, prompt, more)) or parsed
    return {n: o for n, o in (parsed.get("attributes") or {}).items()
            if isinstance(o, dict) and n in names}


def _build_label(seg, cat, score, trajectory, matched, defs, rule_node,
                 idx, selected, extracted) -> Label:
    """STORE: one Label per attribute-driven category. `evidence_attributes`
    covers EVERY defined attribute — selected+extracted ones carry their value +
    evidence (source judge/extract); unselected/empty ones are stored with an
    EMPTY value (value="", evidence=None, source judge/unselected) so
    considered-and-empty is distinguishable downstream. Cited pins = all attr-def
    nodes + the rule node; tool_trace holds the decision trajectory; rationale is
    the matched-rule note (as today)."""
    evidence: list[Attribute] = []
    for name in idx["names"]:
        d = idx["def_by_name"][name]
        o = extracted.get(name)
        if name in selected and o is not None:
            evidence.append(Attribute(
                key=name, value=_attr_value(o.get("value")),
                layer=AttributeLayer.POLICY, source="judge/extract",
                evidence=str(o.get("evidence") or "") or None,
                policy_version=d.version))
        else:
            evidence.append(Attribute(
                key=name, value="", layer=AttributeLayer.POLICY,
                source="judge/unselected", evidence=None,
                policy_version=d.version))

    # cited pins: every attribute-def node + the rule node, canonicalised.
    ver_map = {d.policy_id: d.version for d in defs}
    ver_map[rule_node.policy_id] = rule_node.version
    cited = _normalise_pins(
        [d.policy_id for d in defs] + [rule_node.policy_id], ver_map)

    if matched is not None:
        rationale = (f"[{cat}] rule #{trajectory['rule_index']} matched "
                     f"(score {score}): {trajectory['rule_note']}").rstrip()
    else:
        rationale = (f"[{cat}] no decision rule matched the extracted attributes; "
                     f"applied default score {score}.")
    return Label(
        label_id="", segment_id=seg.segment_id, category=cat, score=score,
        rationale=rationale, cited_policy_ids=cited, evidence_attributes=evidence,
        tool_trace=[{"decision": trajectory}])


def _judge_pipeline(orch, state, seg, asr, frames, attr_cats, cat_defs, cat_rule):
    """SELECT -> EXTRACT -> DECIDE(+trajectory) -> REVIEW -> STORE across the
    attribute-driven categories for one shot. Frames are sampled once by the
    caller and reused across the SELECT / EXTRACT / REVIEW calls. Returns
    (labels, rule-change proposals)."""
    idx = {c: _attr_index(cat_defs[c]) for c in attr_cats}
    names_by_cat = {c: idx[c]["names"] for c in attr_cats}

    # 1. SELECT — one call across all attribute-driven categories: which
    # categories are relevant and, per category, which attributes to label.
    sel_parsed = _parse_json(orch.judge(
        _SELECT_SYS, _select_prompt(state, seg, asr, names_by_cat), frames))
    selected = _parse_selections(sel_parsed, attr_cats, names_by_cat)

    # 2. EXTRACT (per selected category) + 3. DECIDE (deterministic tree over the
    # extracted values). A category with no selected attributes skips its EXTRACT
    # call; the empty-safe tree then falls through to its default score.
    extracted_by_cat: dict[str, dict] = {}
    scores: dict[str, int] = {}
    matched_by_cat: dict[str, object] = {}
    trajectories: dict[str, dict] = {}
    for cat in attr_cats:
        names = selected[cat]
        extracted = ({} if not names
                     else _extract_selected(orch, state, seg, asr, frames, cat,
                                            idx[cat], names))
        extracted_by_cat[cat] = extracted
        values = {n: o.get("value") for n, o in extracted.items()}

        tree = cat_rule[cat].structured_data or {}
        rules = tree.get("rules") or []
        score, matched = _apply_decision_tree(
            rules, tree.get("default", 0), values, idx[cat]["order"])
        scores[cat] = score
        matched_by_cat[cat] = matched
        trajectories[cat] = {
            "selected": list(names),
            "extracted": dict(values),
            "rule_index": rules.index(matched) if matched is not None else None,
            "rule_note": matched.get("note", "") if matched is not None else "",
            "score": score,
        }

    # 4. REVIEW — one call across the categories; the model judges appropriateness
    # but CANNOT change the deterministic score. needs_change -> rule-change.
    items = [{"category": c, "score": scores[c], "trajectory": trajectories[c]}
             for c in attr_cats]
    rev_parsed = _parse_json(orch.judge(
        _REVIEW_SYS, _review_prompt(state, seg, asr, items), frames))
    reviews = _parse_reviews(rev_parsed, attr_cats)

    # 5. STORE — one Label per attribute-driven category; map a flagged review to
    # the rule-change proposal shape _side_fx already consumes.
    labels: list[Label] = []
    proposals: list[dict] = []
    for cat in attr_cats:
        labels.append(_build_label(
            seg, cat, scores[cat], trajectories[cat], matched_by_cat[cat],
            cat_defs[cat], cat_rule[cat], idx[cat], selected[cat],
            extracted_by_cat[cat]))
        rv = reviews.get(cat)
        if rv and rv["needs_change"]:
            note = rv["change_note"]
            proposals.append({"rule_change": {
                "category": cat, "target_policy_id": cat_rule[cat].policy_id,
                "segment_id": seg.segment_id,
                "change": (f"Adjust the {cat} decision tree: {note}" if note
                           else f"Adjust the {cat} decision tree to fit this shot."),
                "rationale": (f"Reviewer flagged shot {seg.segment_id} score "
                              f"{scores[cat]} as inappropriate for the extracted "
                              f"attributes {trajectories[cat]['extracted']}."),
            }})
    return labels, proposals


def _build_bootstrap_label(seg, cat, score, rationale, cited, defs, idx,
                           extracted) -> Label:
    """STORE for the bootstrap attr path: one Label carrying the FULL extracted
    attribute vector as `evidence_attributes` (like `_build_label`) plus a FREE
    holistic score (no decision tree ran). Every defined attribute is
    represented — extracted ones with value + evidence (source judge/extract),
    ones the model dropped stored EMPTY (source judge/unselected). This label is
    one CART training row: attribute vector -> score."""
    evidence: list[Attribute] = []
    for name in idx["names"]:
        d = idx["def_by_name"][name]
        o = extracted.get(name)
        if o is not None:
            evidence.append(Attribute(
                key=name, value=_attr_value(o.get("value")),
                layer=AttributeLayer.POLICY, source="judge/extract",
                evidence=str(o.get("evidence") or "") or None,
                policy_version=d.version))
        else:
            evidence.append(Attribute(
                key=name, value="", layer=AttributeLayer.POLICY,
                source="judge/unselected", evidence=None,
                policy_version=d.version))
    decision = {"category": cat, "mode": "bootstrap_free", "score": score,
                "extracted": {n: o.get("value") for n, o in extracted.items()}}
    return Label(
        label_id="", segment_id=seg.segment_id, category=cat, score=score,
        rationale=rationale, cited_policy_ids=cited, evidence_attributes=evidence,
        tool_trace=[{"decision": decision}])


def _judge_bootstrap_attrs(orch, state, seg, asr, frames, cats, cat_defs,
                           policies):
    """Bootstrap attribute path: EXTRACT every defined attribute (no SELECT — all
    are extracted) then assign a FREE holistic score per category (one scoring
    call, NOT the decision tree, which does not exist yet). The extracted vector
    + free score become the CART training rows. Returns labels."""
    idx_by_cat = {c: _attr_index(cat_defs[c]) for c in cats}

    # EXTRACT — one call per category over ALL its attributes; display values feed
    # the scoring call for grounding.
    extracted_by_cat: dict[str, dict] = {}
    display: list[Attribute] = []
    for cat in cats:
        idx = idx_by_cat[cat]
        extracted = _extract_selected(
            orch, state, seg, asr, frames, cat, idx, idx["names"])
        extracted_by_cat[cat] = extracted
        for n, o in extracted.items():
            display.append(Attribute(
                key=f"{cat}.{n}", value=_attr_value(o.get("value")),
                layer=AttributeLayer.POLICY, source="judge/extract",
                evidence=str(o.get("evidence") or "") or None))

    # FREE holistic score across these categories, grounded in the extracted
    # attributes (shown as policy_attributes). No precedents exist in bootstrap.
    system = _JUDGE_SYS.format(cats=", ".join(cats))
    policy_text = "\n".join(
        f"- ({p.policy_id},v{p.version}) [{p.type.value}/{p.category.value}] {p.text}"
        for p in policies if p.category.value in cats
    ) or "none"
    version_by_id = {p.policy_id: p.version for p in policies}
    prompt = _judge_prompt(state, seg, asr, display, policy_text, [])
    parsed = _parse_json(orch.judge(system, prompt, frames))
    if parsed.get("need_more_frames") and frames:
        more = agent_tools.expand_frames(seg, _FRAMES_PER_SHOT * 2)
        parsed = _parse_json(orch.judge(system, prompt, more)) or parsed

    judged: dict[str, dict] = {}
    for j in parsed.get("judgements", []) or []:
        c = j.get("category")
        if c in cats and c not in judged:
            judged[c] = j

    labels: list[Label] = []
    for cat in cats:
        idx = idx_by_cat[cat]
        j = judged.get(cat, {})
        score = _clamp(j.get("score"))
        rationale = (j.get("rationale", "") or "") or (
            f"[{cat}] free bootstrap score {score} (no decision tree yet).")
        cited = _normalise_pins(list(j.get("cited_policy_ids") or []), version_by_id)
        labels.append(_build_bootstrap_label(
            seg, cat, score, rationale, cited, cat_defs[cat], idx,
            extracted_by_cat[cat]))
    return labels


def _judge_holistic(orch, state, seg, asr, frames, cats, policies, precedents,
                    bootstrap, policy_store, retrieval):
    """Holistic multimodal scoring for categories without an attribute tree (and
    for every category in bootstrap). Preserves the legacy DERIVE (structured
    term levels), one JSON call scoring the given cats, bootstrap gap/structured
    proposals, and weak precedent-divergence checking. Returns (labels, props)."""
    system = _JUDGE_SYS.format(cats=", ".join(cats))
    if bootstrap:
        system += _BOOTSTRAP_SUFFIX

    policy_text = "\n".join(
        f"- ({p.policy_id},v{p.version}) [{p.type.value}/{p.category.value}] {p.text}"
        for p in policies if p.category.value in cats
    ) or "none"
    version_by_id = {p.policy_id: p.version for p in policies}

    # DERIVE (legacy): graded policy-layer attributes from structured term lists.
    pattrs: list[Attribute] = []
    for cat in cats:
        for node in policy_store.get_policy_tree(cat):
            sd = node.structured_data or {}
            if sd.get("kind") != "term_levels":
                continue
            matched = retrieval.match_term_levels(sd, asr)
            if not matched:
                continue
            max_level = max(int(lvl) for lvl in matched)
            prefix = f"{cat}.attr."
            name = (node.policy_id[len(prefix):]
                    if node.policy_id.startswith(prefix) else node.policy_id)
            pattrs.append(Attribute(
                key=f"{cat}.{name}_level", value=int(max_level),
                layer=AttributeLayer.POLICY, source="judge/derive",
                evidence=", ".join(matched.get(str(max_level), [])),
                policy_version=node.version))

    prompt = _judge_prompt(state, seg, asr, pattrs, policy_text, precedents)
    parsed = _parse_json(orch.judge(system, prompt, frames))
    if parsed.get("need_more_frames") and frames:
        more = agent_tools.expand_frames(seg, _FRAMES_PER_SHOT * 2)
        parsed = _parse_json(orch.judge(system, prompt, more)) or parsed

    proposals: list[dict] = []
    if bootstrap:
        for g in parsed.get("policy_gaps") or []:
            proposals.append({
                "segment_id": seg.segment_id, "category": g.get("category"),
                "kind": g.get("kind", "edge_case"),
                "suggestion": g.get("suggestion", "") or "",
                "rationale": g.get("rationale", "") or ""})
        for sa in parsed.get("structured_attributes") or []:
            proposals.append({"structured_attribute": {
                "segment_id": seg.segment_id, "category": sa.get("category"),
                "name": sa.get("name"), "levels": sa.get("levels") or {},
                "description": sa.get("description")}})

    seg_prec = [pr for pr in precedents if pr["segment_id"] == seg.segment_id]
    drafts: list[Label] = []
    for j in parsed.get("judgements", []):
        cat = j.get("category")
        if cat not in cats:
            continue
        score = _clamp(j.get("score"))
        rationale = j.get("rationale", "") or ""
        decision = {"category": cat, "mode": "holistic", "score": score}
        prec_scores = [pl["score"] for pr in seg_prec for pl in pr["labels"]
                       if pl.get("category") == cat and pl.get("score") is not None]
        if prec_scores and all(abs(score - ps) >= 2 for ps in prec_scores):
            decision["precedent_divergence"] = prec_scores
            rationale += " (diverges from precedent; retained for human triage)"
        drafts.append(Label(
            label_id="", segment_id=seg.segment_id, category=cat, score=score,
            rationale=rationale, evidence_attributes=pattrs,
            cited_policy_ids=_normalise_pins(
                list(j.get("cited_policy_ids") or []), version_by_id),
            confidence=j.get("confidence"), tool_trace=[{"decision": decision}]))
    return drafts, proposals


def _side_fx(state: LabellingState) -> dict:
    """Side-effect slot: propose_policy_change (queued for human) /
    define_structured_attribute (bootstrap direct upsert). Proposals are
    dispatched by shape: attribute-driven JUDGE enqueues rule-change requests,
    bootstrap enqueues free-text gaps and drafts structured attributes."""
    for p in state.get("proposals", []) or []:
        # Rule-change request from the attribute-driven tree (never auto-applied).
        rc = p.get("rule_change")
        if rc:
            agent_tools.propose_policy_change(
                change=rc["change"], rationale=rc["rationale"],
                affected=[rc["segment_id"]] if rc.get("segment_id") else [],
                category=rc.get("category"), node_type="decision_rule",
                target_policy_id=rc.get("target_policy_id"))
            continue
        # Structured attribute: direct upsert (bootstrap drafting), not queued.
        sa = p.get("structured_attribute")
        if sa:
            if sa.get("category") and sa.get("name") and sa.get("levels"):
                agent_tools.define_structured_attribute(
                    category=sa["category"], name=sa["name"],
                    levels=sa["levels"], description=sa.get("description"))
            continue
        # Free-text bootstrap gap: human-gated via the change-request queue.
        agent_tools.propose_policy_change(
            change=p.get("suggestion", ""), rationale=p.get("rationale", ""),
            affected=[p["segment_id"]] if p.get("segment_id") else [],
            category=p.get("category"), node_type=p.get("kind", "edge_case"))
    return {}


def _commit(state: LabellingState) -> dict:
    """Persist labels, refresh the rolling carry-over, advance the cursor.

    A label's audit trail is its evidence_attributes + cited_policy_ids + the
    compact decision entry set in JUDGE; the per-stage tool_trace dump is no
    longer merged in here."""
    drafts = state.get("draft_labels", [])
    window = state.get("window", [])
    used_ids = [s.segment_id for s in window]

    committed: list[Label] = []
    for lbl in drafts:
        lbl.label_id = lbl.label_id or str(uuid4())
        lbl.used_segment_ids = [i for i in used_ids if i != lbl.segment_id]
        agent_tools.emit_label(lbl)
        committed.append(lbl)

    bits = [f"{lbl.segment_id}:{lbl.category.value}={lbl.score}" for lbl in committed]
    carry = (state.get("carry_over", "") + " | " + ", ".join(bits)).strip(" |")

    cursor = state["cursor"] + state.get("window_stride", 3)
    return {"cursor": cursor, "carry_over": carry, "draft_labels": []}


def _route(state: LabellingState) -> str:
    return "load" if state["cursor"] < len(state["all_segments"]) else END


# --------------------------------------------------------------------------- #
# graph construction / run
# --------------------------------------------------------------------------- #
def build_graph():
    """Compile the StateGraph: one node per stage, with a conditional edge from
    COMMIT that advances the window cursor and loops to LOAD until exhausted."""
    g = StateGraph(LabellingState)
    g.add_node("load", _load)
    g.add_node("retrieve", _retrieve)
    g.add_node("judge", _judge)
    g.add_node("side_fx", _side_fx)
    g.add_node("commit", _commit)

    g.add_edge(START, "load")
    g.add_edge("load", "retrieve")
    g.add_edge("retrieve", "judge")
    g.add_edge("judge", "side_fx")
    g.add_edge("side_fx", "commit")
    g.add_conditional_edges("commit", _route, {"load": "load", END: END})
    return g.compile()


def label_video(video_id: str, bootstrap: bool = False) -> None:
    """Run the compiled graph across one video's shot windows.

    bootstrap=True disables precedent retrieval (no confirmed labels exist yet)
    and lets JUDGE propose policy gaps that SIDE_FX queues for human review."""
    from models import base_config
    from tools import storage

    segments = storage.get_segments(video_id)
    video = storage.get_video(video_id)
    utterances = storage.get_utterances(video_id)
    cfg = base_config().get("labelling", {})
    stride = int(cfg.get("window_stride", 3))

    init: LabellingState = {
        "video_id": video_id,
        "global_overview": (video.global_overview if video else "") or "",
        "bootstrap": bootstrap,
        "all_segments": segments,
        "utterances": utterances,
        "cursor": 0,
        "window_size": int(cfg.get("window_size", 5)),
        "window_stride": stride,
        "carry_over": "",
        "tool_trace": [],
    }
    n_windows = max(1, math.ceil(len(segments) / max(1, stride)))
    graph = build_graph()

    # Langfuse tracing (opt-in): one trace per video, a span per graph node.
    # `handler` is None when tracing is off, so callbacks stays empty.
    from models.tracing import flush, get_langfuse_handler

    handler = get_langfuse_handler()
    config: dict = {"recursion_limit": n_windows * 6 + 10}
    if handler is not None:
        config["callbacks"] = [handler]
        config["run_name"] = f"label_video:{video_id}"
        config["metadata"] = {"video_id": video_id, "bootstrap": bootstrap}
    try:
        graph.invoke(init, config=config)
    finally:
        flush()

    from schemas.enums import VideoStatus
    storage.set_video_status(video_id, VideoStatus.LABELLED.value)
