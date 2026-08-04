"""Shared enums — the controlled vocabulary of the domain."""
from enum import Enum


class Category(str, Enum):
    """PoC moderation categories (PEGI subset)."""

    GAMBLING = "gambling"
    BAD_LANGUAGE = "bad_language"
    SEX = "sex"


class AttributeLayer(str, Enum):
    BASE = "base"      # policy-independent, produced by ingestion
    POLICY = "policy"  # policy-dependent, produced by the labelling agent


class PolicyType(str, Enum):
    """Kinds of policy node within a category tree."""

    SCORING = "scoring"            # score-band rubric (0..5)
    ATTRIBUTE = "attribute"        # definition of a policy attribute
    EDGE_CASE = "edge_case"        # incremental edge-case rule
    DECISION_RULE = "decision_rule"  # attribute-based decision rule tree


class VideoStatus(str, Enum):
    PENDING = "pending"
    INGESTING = "ingesting"
    INGESTED = "ingested"
    LABELLED = "labelled"
    FAILED = "failed"


class ChangeRequestStatus(str, Enum):
    QUEUED = "queued"
    APPROVED = "approved"
    REJECTED = "rejected"


# PEGI-style age scoring: 0:3, 1:7, 2:12, 3:16, 4:18, 5:blocked
SCORE_MIN = 0
SCORE_MAX = 5
