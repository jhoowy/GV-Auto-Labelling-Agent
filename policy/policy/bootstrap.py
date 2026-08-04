"""Bootstrap — establish the initial policy set.

Policy is authored **data-independently** by the policy LLM (`gpt-5.6-sol`): a
category's attribute schema, base decision tree, and refined 0..5 rubric are
designed from the PEGI seed rubric + a reference signal exemplar + the general-
signal design principle alone — no collected video segments are consulted.

    seed(PEGI -> rubric nodes = v0) -> author(rubric + exemplar + principle)
    -> upsert refined rubric + ATTRIBUTE nodes + DECISION_RULE tree per category
"""
from __future__ import annotations

import logging

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
    duplicating). These nodes are the policy-set v0 baseline and the seed
    context the policy LLM refines. No ATTRIBUTE nodes are seeded — the
    attribute schema is authored (data-independently) during bootstrap.
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


def _scoring_text(category: str) -> str:
    """Human-readable scoring rubric for a category, from the live tree."""
    for node in policy_store.get_policy_tree(category):
        if node.type == PolicyType.SCORING:
            return node.text
    return ""


# --- data-independent authoring -------------------------------------------

# Condensed form of the reference project's sexual "signal" schema
# (dt-labeling .../signal_tree/schemas/sexual.yaml). Included as an EXEMPLAR of
# a well-formed GENERAL signal set — its SHAPE, not its content, is the lesson:
# every attribute is a closed-enum observation; specificity lives in the enum
# VALUES and a reused ordered priority/visibility axis, never in a verdict name.
_EXEMPLAR = (
    "Reference exemplar — a general 'signal' schema (from a sexual-content "
    "project), shown ONLY to illustrate well-formed SHAPE, never to copy:\n"
    "- Organised into a few sections (exposure / pose / text / context).\n"
    "- Each section reuses ordered axes — `target` (female/male/ambiguous), "
    "`priority` (low<medium<high), `visibility_level` (unclear<visible<"
    "emphasize) — plus ONE closed `*_names` enum of the specific observable "
    "things (exposure part_names: genital, nipple, cleavage, thigh, ...; pose "
    "action_names: explicit_sex_act, kissing, seductive_gaze, ...).\n"
    "- `ordinals` map each ordered enum value to a rank so rules compare by "
    "DEGREE (priority low=1<medium=2<high=3; visibility unclear=1<visible=2<"
    "emphasize=3).\n"
    "Lesson: specificity lives in the ENUM VALUES + a reused ORDERED axis, not "
    "in many narrow booleans."
)

# What separates a reusable signal from a baked-in verdict.
_DESIGN_PRINCIPLE = (
    "An ATTRIBUTE is a GENERAL, reusable signal a labeller can point at in one "
    "pass — a CLOSED enum (categorical, or ordinal ordered low..high), NEVER a "
    "verdict. Put specificity in the ENUM VALUES of one general attribute, not "
    "in many narrow booleans (NOT is_slot_machine; instead one "
    "`gambling_activity` with values none/simulated_casino/loot_box/"
    "real_money_casino). Ban attribute names containing is_/core/frequent/"
    "functional/real/enough/primary — those are judgments; express them as an "
    "ordinal value or a decision-tree rule."
)

_AUTHOR_SYS = (
    "You are a content-moderation policy engineer. Working DATA-INDEPENDENTLY "
    "(you are shown no example videos), you design ONE category's labelling "
    "policy from first principles: a small set of GENERAL, observable "
    "attributes (signals), an attribute-based decision tree mapping their "
    "values to a 0..5 age score, and a refined 0..5 rubric consistent with "
    "both. Respond with a single JSON object only, no prose."
)


def _author_prompt(cat: str, rubric: str) -> str:
    """The user prompt: base rubric (seed) + exemplar + principle + output spec.
    No video data is included — authoring is by design data-independent."""
    return (
        f"Category: {cat}\n\n"
        f"Base scoring rubric (PEGI-aligned 0..5 age bands) to refine:\n"
        f"{rubric}\n\n"
        f"{_EXEMPLAR}\n\n"
        f"{_DESIGN_PRINCIPLE}\n\n"
        "Design this category's policy WITHOUT reference to any specific video. "
        "Return a single JSON object with EXACTLY these keys:\n"
        '{"rubric": "<refined 0..5 rubric text, one line per band 0..5>",\n'
        ' "attributes": [ {"name":"...", "value_type":'
        '"categorical|ordinal|boolean", "values":[ {"value":"...","label":'
        '"...","description":"...","examples":["..."],"rules":["..."]} ], '
        '"guidelines":"...", "scores_informed":[<ints>]} ],\n'
        ' "decision_tree": {"default":0, "rules":[ {"when":[ {"attribute":'
        '"...","op":"==|>=|<=|in|present","value":<val>} ], "score":<int>, '
        '"note":"..."} ]} }\n\n'
        "Requirements: 4-8 GENERAL attributes; each is a closed-enum "
        "observation (specificity in the values, not the name); ordinal "
        "attributes list values in ASCENDING order low..high so rules compare "
        "with >=; boolean attributes omit `values`; each value carries a "
        "`rules` list of 2-4 concrete edge-case handling notes in the form "
        "'case ... -> this value' (e.g. a slot machine briefly visible in the "
        "background of a non-gambling scene -> incidental) that disambiguate it "
        "from neighbouring values — the labeller reads these when assigning the "
        "value; the decision tree is priority-ordered (highest score first) "
        "over ONLY these attributes and their listed values, with band 0 as the "
        "default; the rubric's bands stay consistent with the attributes and "
        "tree. Output JSON only."
    )


def _get_policy_llm():
    from models import get_policy_llm
    return get_policy_llm()


def _base_rubric(cat: str) -> str:
    """Authoring seed context: the PEGI seed rubric text for a category, falling
    back to the live SCORING node if the category has no PEGI seed."""
    try:
        text = _PEGI_RUBRICS.get(Category(cat))
    except ValueError:
        text = None
    return text or _scoring_text(cat)


def _as_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _as_int_list(v) -> list[int]:
    """Valid 0..5 score bands from a loose list; drops non-ints and negatives."""
    out: list[int] = []
    for x in v if isinstance(v, list) else []:
        i = _as_int(x, -1)
        if i >= 0:
            out.append(i)
    return out


def _as_dicts(v) -> list[dict]:
    """Coerce an attributes/rules field into a list of dicts. Accepts a list, or
    a name-keyed dict (some replies key items by name)."""
    if isinstance(v, list):
        return [x for x in v if isinstance(x, dict)]
    if isinstance(v, dict):
        return [{"name": k, **x} for k, x in v.items() if isinstance(x, dict)]
    return []


def author_category_policy(category, *, policy_llm=None) -> dict:
    """Author a category's policy DATA-INDEPENDENTLY with the policy LLM.

    No video data is consulted. The model (`gpt-5.6-sol`) is given (a) the
    category's base PEGI rubric, (b) the reference sexual signal schema as a
    condensed exemplar of well-formed general signals, and (c) the general-
    signal design principle, and asked in ONE JSON call to design: a refined
    0..5 rubric, 4-8 general closed-enum attributes, and a base decision tree
    over exactly those attributes. Each piece is applied to the tree — the
    refined rubric to the `{cat}.scoring` node, each attribute to a `{cat}.attr.
    <name>` ATTRIBUTE node, and the tree to the `{cat}.rules` DECISION_RULE
    node. Lenient about the reply shape. Idempotent: upsert bumps node
    versions, so re-running re-authors in place. Returns a summary."""
    cat = getattr(category, "value", category)
    rubric = _base_rubric(cat)
    llm = policy_llm or _get_policy_llm()
    result = llm.complete_json(_AUTHOR_SYS, _author_prompt(cat, rubric))

    # Refined rubric -> the SCORING node (fall back to the base rubric).
    new_rubric = (result.get("rubric") or "").strip() or rubric
    policy_store.upsert_policy(Policy(
        policy_id=f"{cat}.scoring",
        type=PolicyType.SCORING,
        category=Category(cat),
        text=new_rubric,
    ))

    # Attribute definitions -> one ATTRIBUTE node each.
    names: list[str] = []
    for a in _as_dicts(result.get("attributes")):
        name = str(a.get("name") or "").strip()
        if not name:
            continue
        policy_store.upsert_attribute_definition(
            cat, name,
            value_type=a.get("value_type") or a.get("type") or "categorical",
            guidelines=a.get("guidelines") or a.get("description") or "",
            scores_informed=_as_int_list(a.get("scores_informed")),
            values=a.get("values") or a.get("enum"),
        )
        names.append(name)

    # Base decision tree -> the DECISION_RULE node.
    tree = result.get("decision_tree") or result.get("tree") or {}
    rules = [r for r in _as_dicts(tree.get("rules")) if r.get("when") is not None]
    default = _as_int(tree.get("default"), 0)
    policy_store.upsert_decision_rule(cat, rules, default)

    log.info(
        "bootstrap: authored %d attribute(s) + %d rule(s) for %s",
        len(names), len(rules), cat,
    )
    return {
        "category": cat,
        "attributes": names,
        "n_rules": len(rules),
        "default": default,
    }


def run_bootstrap(video_ids: list[str] | None = None) -> dict:
    """Author the initial policy set DATA-INDEPENDENTLY for every active category.

    No video data is consulted: each category's attribute schema, base decision
    tree, and refined rubric are designed by the policy LLM (`gpt-5.6-sol`) from
    the PEGI seed rubric + the reference signal exemplar + the general-signal
    design principle alone. `video_ids` is accepted but ignored (kept for
    call-site compatibility).

    Flow:
      1. SEED: ensure PEGI v0 scoring rubrics exist (the authoring seed).
      2. AUTHOR: per active category, `author_category_policy` designs and
         upserts the refined rubric + ATTRIBUTE nodes + DECISION_RULE tree.
    Idempotent: upsert bumps each node's version on re-run.
    """
    from models import base_config

    seed_from_pegi()
    categories = base_config().get("policy", {}).get("categories", [])
    authored: dict = {}
    for cat in categories:
        try:
            authored[cat] = author_category_policy(cat)
        except Exception as e:  # noqa: BLE001 - one category must not abort the run
            log.warning("bootstrap: authoring failed for %s (%s)", cat, e)
    return {
        "seeded_categories": [c.value for c in _PEGI_RUBRICS],
        "authored": authored,
    }
