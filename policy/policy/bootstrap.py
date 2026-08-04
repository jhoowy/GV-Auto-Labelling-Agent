"""Bootstrap — establish the initial policy set.

Reuses the normal mechanism (labelling agent + propose_policy_change queue),
run intensively over bootstrap videos, then converged.

    seed(PEGI -> rubric nodes = v0) -> label bootstrap videos -> cluster
    proposals into edge-case candidates -> human review -> iterate until
    cross-sample label variance converges -> policy-set v1
"""
from __future__ import annotations

import logging
from pathlib import Path

from schemas import Category, Policy
from schemas.enums import PolicyType
from tools import policy_store

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]


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
    duplicating). These nodes are the policy-set v0 baseline.
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

    # Structured data (profanity word list) lives outside the human-readable
    # rubric, attached to a bad_language ATTRIBUTE node via structured_ref so the
    # agent's DERIVE step can match ASR against it.
    if Category.BAD_LANGUAGE in categories:
        wordlist = _ROOT / "data" / "wordlists" / "bad_language.txt"
        policy_store.upsert_policy(Policy(
            policy_id="bad_language.profanity_list",
            type=PolicyType.ATTRIBUTE,
            category=Category.BAD_LANGUAGE,
            parent_id="bad_language.scoring",
            text=("Profanity attribute: presence of profanity or slurs in the "
                  "shot's ASR is a signal for the bad_language score. The terms "
                  "live in an external word list (structured_ref), not here."),
            structured_ref=str(wordlist),
        ))
        log.info("seeded bad_language profanity_list ATTRIBUTE node")


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
      3. CLUSTER: group queued proposals into candidate rules.
      4. REVIEW (external, human-gated): approve/reject via resolve_change_request;
         approved candidates become nodes and the tree is snapshotted as v1.
         Nothing here auto-applies changes or auto-snapshots.
    """
    from labelling import label_video

    seed_from_pegi()

    for vid in video_ids:
        log.info("bootstrap: labelling %s (retrieval disabled)", vid)
        label_video(vid, bootstrap=True)

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
        "n_proposals": len(proposals),
        "n_candidates": len(clusters),
        "candidates": [
            {"size": len(c), "example": c[0].proposed_change if c else ""}
            for c in clusters
        ],
    }
