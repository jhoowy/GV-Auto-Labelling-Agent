"""LLM-authored EN + KO descriptions for a category's decision-tree rules.

After CART learns a category's `{category}.rules` node (`tree_train`), each rule
is machine-shaped — `{"when":[...],"score":int,"note":"CART leaf (n=..)"}` —
legible to `_apply_decision_tree` but opaque to a human reviewer. Here ONE
`get_policy_llm().complete_json` call turns the ordered rules (their conditions +
assigned score) plus the attribute DEFINITIONS (which give each value its meaning)
into a short TITLE-LEVEL label (~3-6 words) plus a full sentence, each with a
Korean translation, per rule. These are merged back onto the rule dicts BY INDEX
and the node is re-saved via `upsert_decision_rule` — the storage format is
unchanged (`_apply_decision_tree` ignores the extra
`title`/`title_ko`/`description`/`description_ko` keys); they are pure
presentation.

The index-merge (`_merge_descriptions`) is a pure function so it is unit-testable
without a DB or a live LLM. Descriptions are best-effort: `train_decision_tree`
calls `describe_rules` guarded, so a missing API key never fails training.
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)

_SYSTEM = (
    "You document a content-moderation decision tree for gameplay video. You are "
    "given a category's ordered rules (each a conjunction of attribute conditions "
    "plus an assigned age score 0-5) and the attribute DEFINITIONS that give each "
    "value its meaning, plus a fallthrough default score. For EACH rule write: "
    "(a) a short TITLE-LEVEL label of ~3-6 words naming the content the rule "
    'captures (e.g. "No gambling", "Incidental no-stake play", "Explicit betting '
    'solicitation"), and its natural Korean translation; and (b) one short '
    "concrete sentence describing that content and why it earns its score, and a "
    "natural Korean translation of that sentence. Titles are terse labels, "
    "descriptions are full sentences. Also title + describe the default "
    "(no-rule-matches) case. Return JSON exactly as: "
    '{"rules":[{"index":<int>,"title":<en>,"title_ko":<ko>,'
    '"description":<en>,"description_ko":<ko>}, ...], '
    '"default_title":<en>,"default_title_ko":<ko>,'
    '"default_description":<en>,"default_description_ko":<ko>} with one rules '
    "entry per input rule, index-aligned to the input order."
)


# --------------------------------------------------------------------------- #
# prompt shaping
# --------------------------------------------------------------------------- #
def _attr_context(defs) -> list[dict]:
    """Compact attribute-def context for the prompt: each attribute's name, type,
    detection guidelines and its value meanings (value -> label + description) so
    the author can read what a condition like `activity in [real]` actually means."""
    out: list[dict] = []
    for d in defs:
        sd = d.structured_data or {}
        if sd.get("kind") != "attribute_def":
            continue
        name = d.policy_id.split(".attr.", 1)[1] if ".attr." in d.policy_id else d.policy_id
        values = []
        for v in sd.get("values") or []:
            if isinstance(v, dict):
                values.append({"value": v.get("value"), "label": v.get("label"),
                               "description": v.get("description")})
            else:
                values.append({"value": v})
        out.append({"attribute": name, "value_type": sd.get("value_type"),
                    "guidelines": sd.get("guidelines"), "values": values})
    return out


def _rules_for_prompt(rules) -> list[dict]:
    """The rules reduced to what the author needs: index, conditions, score."""
    return [{"index": i, "when": r.get("when", []), "score": r.get("score")}
            for i, r in enumerate(rules)]


# --------------------------------------------------------------------------- #
# merge (pure, unit-tested without a DB or an LLM)
# --------------------------------------------------------------------------- #
def _clean(s):
    """A non-empty stripped string, else None."""
    return s.strip() if isinstance(s, str) and s.strip() else None


def _merge_descriptions(rules, response) -> list[dict]:
    """PURE: merge an LLM response's per-index EN/KO title + description onto `rules`.

    `response` is
    `{"rules":[{"index","title","title_ko","description","description_ko"}, ...]}`.
    Each entry sets any of `title`/`title_ko`/`description`/`description_ko` on
    `rules[index]` when `index` is a real in-range position; entries that are
    malformed, out of range, or carry blank text (per field) are ignored, and
    rules the response omits keep their prior state — so a short / long /
    misaligned response can only ADD aligned fields, never corrupt the tree.
    Returns a NEW list of shallow-copied rule dicts; the `{when,score,note}`
    storage keys are untouched."""
    out = [dict(r) for r in rules]
    entries = response.get("rules") if isinstance(response, dict) else None
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        idx = e.get("index")
        if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < len(out):
            continue
        # each field set independently; a blank one is skipped, not invented.
        for key in ("title", "title_ko", "description", "description_ko"):
            if (val := _clean(e.get(key))) is not None:
                out[idx][key] = val
    return out


# --------------------------------------------------------------------------- #
# describe (DB-backed)
# --------------------------------------------------------------------------- #
def describe_rules(category, llm=None) -> dict:
    """Author EN + KO descriptions for a category's learned decision-tree rules.

    Loads the `{category}.rules` node and the category's attribute defs, makes ONE
    `get_policy_llm().complete_json` call to get a per-rule EN sentence + KO
    translation (plus a default-case description), merges them onto the rules by
    index, and re-saves via `upsert_decision_rule` (storage format unchanged).
    Skips (logs, no write) when the category has no rules node or no rules.
    Returns a summary dict with a `status`."""
    from tools import policy_store

    cat = getattr(category, "value", category)
    tree = policy_store.get_policy_tree(cat)
    node = next((p for p in tree if p.policy_id == f"{cat}.rules"), None)
    if node is None:
        log.info("describe_rules: %s has no decision-tree node; skipping", cat)
        return {"category": cat, "status": "no_tree"}

    sd = node.structured_data or {}
    rules = list(sd.get("rules") or [])
    default = int(sd.get("default", 0))
    if not rules:
        log.info("describe_rules: %s tree has no rules; skipping", cat)
        return {"category": cat, "status": "no_rules"}

    defs = [p for p in tree if (p.structured_data or {}).get("kind") == "attribute_def"]

    if llm is None:
        from models import get_policy_llm

        llm = get_policy_llm()
    user = json.dumps({
        "category": cat,
        "default_score": default,
        "attributes": _attr_context(defs),
        "rules": _rules_for_prompt(rules),
    }, ensure_ascii=False)
    response = llm.complete_json(_SYSTEM, user)

    described = _merge_descriptions(rules, response)
    resp = response if isinstance(response, dict) else {}
    dt = _clean(resp.get("default_title"))
    dt_ko = _clean(resp.get("default_title_ko"))
    dd = _clean(resp.get("default_description"))
    dd_ko = _clean(resp.get("default_description_ko"))
    policy_store.upsert_decision_rule(
        cat, described, default,
        default_title=dt, default_title_ko=dt_ko,
        default_description=dd, default_description_ko=dd_ko)

    n_desc = sum(1 for r in described if r.get("description"))
    log.info("describe_rules: authored %d/%d rule description(s) for %s",
             n_desc, len(described), cat)
    return {"category": cat, "status": "described",
            "n_rules": len(described), "n_described": n_desc,
            "default_described": bool(dd)}


def describe_all(categories=None) -> dict:
    """Author descriptions for every active category's decision tree (or a subset).

    Categories default to the policy config's active set. One category's failure
    is logged and recorded, never aborting the run. Returns {category: summary}."""
    if categories is None:
        from schemas.enums import Category

        from models import base_config
        categories = ((base_config().get("policy", {}) or {}).get("categories")
                      or [c.value for c in Category])
    out: dict = {}
    for c in categories:
        cat = getattr(c, "value", c)
        try:
            out[cat] = describe_rules(cat)
        except Exception as e:  # noqa: BLE001 - one category must not abort the run
            log.warning("describe_rules: failed for %s (%s)", cat, e)
            out[cat] = {"category": cat, "status": "error", "error": str(e)}
    return out
