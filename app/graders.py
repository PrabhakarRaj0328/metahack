"""
Graders for all three Email Triage tasks.

Each grader takes the agent's TriageAction and the gold-label email dict
and returns a StepReward with fine-grained partial credit.

Reward design philosophy:
- Priority: exact=1.0, adjacent=0.5, 2-away=0.2, else=0.0
- Category: exact=1.0, related=0.4, wrong=0.0
- Action: exact=1.0, acceptable_alt=0.6, neutral=0.2, destructive=0.0
- Reply (hard task only): keyword coverage + length heuristic
- Penalties: reply to spam (-0.3), delete legal/urgent (-0.5), no reply when required (-0.2)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional

from app.models import StepReward, TriageAction
from data.emails import priority_partial_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_PRIORITIES = {"urgent", "high", "normal", "low", "spam"}
VALID_CATEGORIES = {
    "support", "billing", "sales", "internal", "legal",
    "hr", "it", "complaint", "inquiry", "spam", "other",
}
VALID_ACTIONS = {"reply", "forward", "archive", "delete", "escalate", "flag", "snooze"}

# Category groups: categories in the same group get 0.4 partial credit
CATEGORY_GROUPS = [
    {"legal", "complaint"},
    {"support", "inquiry"},
    {"billing", "sales"},
    {"hr", "internal"},
    {"it", "internal"},
    {"spam", "other"},
]

# Action acceptability matrix: acceptable_alternatives get 0.6
ACTION_ALTERNATIVES: Dict[str, List[str]] = {
    "escalate": ["forward", "flag"],
    "forward":  ["escalate"],
    "reply":    ["forward"],
    "archive":  ["snooze"],
    "delete":   ["archive"],
    "flag":     ["escalate", "snooze"],
    "snooze":   ["archive", "flag"],
}

# Actions considered destructive when gold is urgent/high
DESTRUCTIVE_FOR_HIGH = {"delete", "archive", "snooze"}


def category_score(predicted: str, gold: str) -> float:
    if predicted == gold:
        return 1.0
    for group in CATEGORY_GROUPS:
        if predicted in group and gold in group:
            return 0.4
    return 0.0


def action_score(predicted: str, gold: str, gold_priority: str) -> tuple[float, Optional[str]]:
    """Returns (score, penalty_reason_or_None)."""
    penalty = None
    if predicted == gold:
        return 1.0, None

    # Destructive action on urgent/high email
    if gold_priority in ("urgent", "high") and predicted in DESTRUCTIVE_FOR_HIGH:
        penalty = f"Destructive action '{predicted}' on {gold_priority}-priority email"
        return 0.0, penalty

    # Acceptable alternative
    if predicted in ACTION_ALTERNATIVES.get(gold, []):
        return 0.6, None

    # Neutral but not great
    return 0.2, None


def reply_quality_score(reply_draft: Optional[str], required_keywords: List[str]) -> float:
    """Score reply quality by keyword coverage + basic length check."""
    if not reply_draft:
        return 0.0
    if not required_keywords:
        # Not required but provided — give neutral credit
        return 0.5

    text = reply_draft.lower()
    hits = sum(1 for kw in required_keywords if kw.lower() in text)
    keyword_ratio = hits / len(required_keywords)

    # Length bonus: replies under 20 chars are useless, over 50 chars are good
    length_score = min(1.0, max(0.0, (len(reply_draft) - 20) / 200))

    return round(0.7 * keyword_ratio + 0.3 * length_score, 3)


# ---------------------------------------------------------------------------
# Task Easy grader — priority classification only
# ---------------------------------------------------------------------------

def grade_easy(action: TriageAction, email: Dict[str, Any]) -> StepReward:
    """
    Easy task: only priority matters.
    Other fields are still evaluated for feedback but don't affect reward.
    """
    gold = email["gold"]
    penalties = []
    feedback_parts = []

    p_score = priority_partial_score(action.priority, gold["priority"])
    feedback_parts.append(
        f"Priority: '{action.priority}' vs gold '{gold['priority']}' → {p_score:.2f}"
    )

    # Penalty: replying to spam
    if gold["priority"] == "spam" and action.action == "reply":
        penalties.append("Replying to spam email")
        p_score = max(0.0, p_score - 0.3)

    reward = round(p_score, 3)
    return StepReward(
        reward=reward,
        priority_score=p_score,
        category_score=0.0,  # not graded this task
        action_score=0.0,    # not graded this task
        reply_score=None,
        feedback=" | ".join(feedback_parts),
        penalties=penalties,
    )


# ---------------------------------------------------------------------------
# Task Medium grader — priority + category + action
# ---------------------------------------------------------------------------

def grade_medium(action: TriageAction, email: Dict[str, Any]) -> StepReward:
    """
    Medium task: priority (40%) + category (30%) + action (30%).
    """
    gold = email["gold"]
    penalties = []
    feedback_parts = []

    p_score = priority_partial_score(action.priority, gold["priority"])
    feedback_parts.append(f"Priority: '{action.priority}' vs '{gold['priority']}' → {p_score:.2f}")

    c_score = category_score(action.category, gold["category"])
    feedback_parts.append(f"Category: '{action.category}' vs '{gold['category']}' → {c_score:.2f}")

    a_score, penalty = action_score(action.action, gold["action"], gold["priority"])
    if penalty:
        penalties.append(penalty)
    feedback_parts.append(f"Action: '{action.action}' vs '{gold['action']}' → {a_score:.2f}")

    # Penalty: delete on legal/urgent
    if gold["category"] == "legal" and action.action == "delete":
        penalties.append("Deleting legal email")
        a_score = 0.0

    # Penalty: replying to spam
    if gold["category"] == "spam" and action.action == "reply":
        penalties.append("Replying to spam")
        p_score = max(0.0, p_score - 0.3)

    reward = round(0.40 * p_score + 0.30 * c_score + 0.30 * a_score, 3)
    return StepReward(
        reward=reward,
        priority_score=p_score,
        category_score=c_score,
        action_score=a_score,
        reply_score=None,
        feedback=" | ".join(feedback_parts),
        penalties=penalties,
    )


# ---------------------------------------------------------------------------
# Task Hard grader — full triage + reply quality
# ---------------------------------------------------------------------------

def grade_hard(action: TriageAction, email: Dict[str, Any]) -> StepReward:
    """
    Hard task: priority (25%) + category (20%) + action (25%) + reply (30%).
    Reply score only applies when reply_required=True.
    """
    gold = email["gold"]
    penalties = []
    feedback_parts = []

    p_score = priority_partial_score(action.priority, gold["priority"])
    feedback_parts.append(f"Priority: '{action.priority}' vs '{gold['priority']}' → {p_score:.2f}")

    c_score = category_score(action.category, gold["category"])
    feedback_parts.append(f"Category: '{action.category}' vs '{gold['category']}' → {c_score:.2f}")

    a_score, penalty = action_score(action.action, gold["action"], gold["priority"])
    if penalty:
        penalties.append(penalty)
    feedback_parts.append(f"Action: '{action.action}' vs '{gold['action']}' → {a_score:.2f}")

    # Phishing detection bonus: marking a phishing email as spam
    is_phishing = gold.get("is_phishing", False)
    if is_phishing and action.priority == "spam" and action.action == "delete":
        p_score = 1.0
        a_score = 1.0
        feedback_parts.append("PHISHING correctly identified")

    # Reply grading
    reply_required = gold.get("reply_required", False)
    required_kws = gold.get("reply_keywords", [])

    if reply_required:
        if action.action != "reply":
            penalties.append("Reply required but action was not 'reply'")
            r_score = 0.0
        else:
            r_score = reply_quality_score(action.reply_draft, required_kws)
        feedback_parts.append(f"Reply quality → {r_score:.2f}")
    else:
        r_score = None
        # Penalty: drafting a reply for an escalate/legal situation
        if gold["action"] in ("escalate", "delete") and action.reply_draft:
            penalties.append("Reply drafted for an email that should be escalated/deleted — policy risk")

    # Penalty: delete on legal/urgent
    if gold["category"] == "legal" and action.action == "delete":
        penalties.append("Deleting legal email")
        a_score = 0.0

    # Penalty: replying to spam/phishing
    if gold.get("is_phishing") and action.action == "reply":
        penalties.append("Replying to phishing email")
        a_score = 0.0
        p_score = max(0.0, p_score - 0.5)

    if r_score is not None:
        reward = round(0.25 * p_score + 0.20 * c_score + 0.25 * a_score + 0.30 * r_score, 3)
    else:
        # No reply required: redistribute reply weight
        reward = round(0.30 * p_score + 0.25 * c_score + 0.45 * a_score, 3)

    # Apply penalty deductions (capped at 0)
    total_penalty = len(penalties) * 0.1
    reward = round(max(0.0, reward - total_penalty), 3)

    return StepReward(
        reward=reward,
        priority_score=p_score,
        category_score=c_score,
        action_score=a_score,
        reply_score=r_score,
        feedback=" | ".join(feedback_parts),
        penalties=penalties,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

GRADERS = {
    "task_easy":   grade_easy,
    "task_medium": grade_medium,
    "task_hard":   grade_hard,
}


def grade(task_id: str, action: TriageAction, email: Dict[str, Any]) -> StepReward:
    grader = GRADERS.get(task_id)
    if grader is None:
        raise ValueError(f"Unknown task_id: {task_id}")
    return grader(action, email)