"""Presentation-layer Korean translation of policy nodes (#22).

English is authoritative for scoring/extraction; this adds a PARALLEL Korean
rendering of only the human-readable strings, stored under
`structured_data.i18n.ko` MIRRORING the English field structure. The English
source is never overwritten — the UI reads Korean when present and falls back to
English otherwise.

Only two node kinds carry translatable prose:
  - `attribute_def` : `guidelines` + each value's `label` / `description` / `rules[]`
  - `decision_tree` : each rule's `note` (index-aligned) + an optional `default_note`
Value keys, attribute names, ops, numbers and structure stay in English.

Translation is cached per node version via `i18n.ko._src_version`: a node is
re-translated only when its version differs, so this is idempotent and cheap to
re-run after a policy edit.

The pure field-mapping helpers (`extract_translatable`, `merge_translation`,
`is_up_to_date`) are DB-free and unit-tested; `translate_*` add the LLM call and
persist via `policy_store.update_structured_data` (no version bump).
"""
from __future__ import annotations

import copy
import json
from typing import Any

from schemas import Policy
from tools import policy_store

_KO = "ko"

_SYS = (
    "You are a professional Korean localiser for a content-moderation policy UI. "
    "Translate the English policy strings in the given JSON into natural, "
    "professional Korean for human reviewers. Translate ONLY the string values; "
    "keep every JSON key, the object/array structure, value identifiers, numbers "
    "and any code-like tokens EXACTLY as given, and preserve list lengths. "
    "Return a single JSON object with the same shape."
)


# --- pure field mapping (DB-free, unit-tested) ----------------------------

def _ko_of(sd: Any) -> dict | None:
    """The stored Korean payload of a node, if any."""
    if isinstance(sd, dict) and isinstance(sd.get("i18n"), dict):
        ko = sd["i18n"].get(_KO)
        return ko if isinstance(ko, dict) else None
    return None


def extract_translatable(sd: Any) -> dict | None:
    """The human-readable strings of a node, in the `i18n.ko` mirror shape.

    Returns None when the node carries nothing to translate (scoring text-only,
    `term_levels` word lists, or an empty payload). Value keys, attribute names,
    ops, numbers and structure are deliberately left out — they stay English.
    """
    if not isinstance(sd, dict):
        return None
    kind = sd.get("kind")

    if kind == "attribute_def":
        out: dict[str, Any] = {}
        g = sd.get("guidelines")
        if isinstance(g, str) and g.strip():
            out["guidelines"] = g
        values: dict[str, dict] = {}
        for v in sd.get("values") or []:
            if not isinstance(v, dict):
                continue
            key = str(v.get("value", v.get("label", "")))
            if not key:
                continue
            fields: dict[str, Any] = {}
            for f in ("label", "description"):
                s = v.get(f)
                if isinstance(s, str) and s.strip():
                    fields[f] = s
            rules = [str(r) for r in (v.get("rules") or []) if str(r).strip()]
            if rules:
                fields["rules"] = rules
            if fields:
                values[key] = fields
        if values:
            out["values"] = values
        return out or None

    if kind == "decision_tree":
        out = {}
        rules_out: list[dict] = []
        any_note = False
        for r in sd.get("rules") or []:
            note = r.get("note") if isinstance(r, dict) else None
            if isinstance(note, str) and note.strip():
                rules_out.append({"note": note})
                any_note = True
            else:
                rules_out.append({})  # placeholder keeps index alignment
        if any_note:
            out["rules"] = rules_out
        dn = sd.get("default_note")
        if isinstance(dn, str) and dn.strip():
            out["default_note"] = dn
        return out or None

    return None


def merge_translation(sd: dict, ko: dict, src_version: int) -> dict:
    """Merge a Korean translation `ko` into a copy of `sd` under `i18n.ko`.

    Mirrors the source structure and preserves ALL English. Only fields present
    in the English source are kept (guards against a model inventing keys or
    dropping list items); the version cache key `_src_version` is stamped so
    re-translation is skipped until the node changes. Returns a new dict.
    """
    src = extract_translatable(sd) or {}
    ko = ko if isinstance(ko, dict) else {}
    out_ko: dict[str, Any] = {"_src_version": int(src_version)}

    if "guidelines" in src and isinstance(ko.get("guidelines"), str):
        out_ko["guidelines"] = ko["guidelines"]

    if "values" in src:
        ko_vals = ko.get("values") if isinstance(ko.get("values"), dict) else {}
        vals_ko: dict[str, dict] = {}
        for key, fields in src["values"].items():
            kf = ko_vals.get(key) if isinstance(ko_vals.get(key), dict) else {}
            merged: dict[str, Any] = {}
            for f in ("label", "description"):
                if f in fields and isinstance(kf.get(f), str):
                    merged[f] = kf[f]
            if "rules" in fields:
                kr = kf.get("rules")
                if isinstance(kr, list) and len(kr) == len(fields["rules"]):
                    merged["rules"] = [str(x) for x in kr]
            if merged:
                vals_ko[key] = merged
        if vals_ko:
            out_ko["values"] = vals_ko

    if "rules" in src:
        kr = ko.get("rules")
        rules_ko: list[dict] = []
        if isinstance(kr, list) and len(kr) == len(src["rules"]):
            for s_rule, k_rule in zip(src["rules"], kr):
                if (s_rule.get("note") and isinstance(k_rule, dict)
                        and isinstance(k_rule.get("note"), str)):
                    rules_ko.append({"note": k_rule["note"]})
                else:
                    rules_ko.append({})
        if any(r for r in rules_ko):
            out_ko["rules"] = rules_ko

    if "default_note" in src and isinstance(ko.get("default_note"), str):
        out_ko["default_note"] = ko["default_note"]

    new_sd = copy.deepcopy(sd)
    i18n = dict(new_sd["i18n"]) if isinstance(new_sd.get("i18n"), dict) else {}
    i18n[_KO] = out_ko
    new_sd["i18n"] = i18n
    return new_sd


def is_up_to_date(sd: Any, version: int) -> bool:
    """True when a current Korean translation already exists for this version."""
    ko = _ko_of(sd)
    return isinstance(ko, dict) and ko.get("_src_version") == version


def needs_translation(node: Policy) -> bool:
    """A node needs (re)translation when it has translatable prose and its stored
    Korean payload is missing or stamped for an older version."""
    sd = node.structured_data or {}
    return extract_translatable(sd) is not None and not is_up_to_date(sd, node.version)


# --- LLM-backed translation + persistence ---------------------------------

def _get_translator():
    from models import get_translation_llm

    return get_translation_llm()


def translate_policy_node(node: Policy, *, translator=None, persist: bool = True) -> Policy | None:
    """Translate one node's human-readable text to Korean and (by default)
    persist it under `structured_data.i18n.ko` WITHOUT bumping the node version.

    Returns the updated Policy, or None when the node has nothing to translate or
    is already current for its version.
    """
    sd = node.structured_data or {}
    if not needs_translation(node):
        return None
    src = extract_translatable(sd)
    llm = translator or _get_translator()
    ko = llm.complete_json(_SYS, json.dumps(src, ensure_ascii=False))
    new_sd = merge_translation(sd, ko, node.version)
    if persist:
        return policy_store.update_structured_data(node.policy_id, new_sd)
    return node.model_copy(update={"structured_data": new_sd})


def translate_category(category: str, *, translator=None) -> dict:
    """Translate every human-readable node of a category into Korean, skipping
    nodes already current for their version. Returns a small summary dict."""
    llm = translator or _get_translator()
    translated: list[str] = []
    skipped: list[str] = []
    for node in policy_store.get_policy_tree(category):
        if extract_translatable(node.structured_data or {}) is None:
            continue  # nothing translatable (scoring text-only / term_levels)
        if not needs_translation(node):
            skipped.append(node.policy_id)
            continue
        translate_policy_node(node, translator=llm)
        translated.append(node.policy_id)
    return {
        "category": category,
        "translated": translated,
        "skipped": skipped,
        "n_translated": len(translated),
        "n_skipped": len(skipped),
    }
