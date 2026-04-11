"""
Typed Pydantic models for the Email Triage OpenEnv environment.
Defines Observation, Action, Reward, and State models.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Observation — what the agent sees each step
# ---------------------------------------------------------------------------
class EmailObservation(BaseModel):
    """Current email the agent must triage, plus inbox context."""
    email_id: str = Field(..., description="Unique email identifier")
    subject: str = Field(..., description="Email subject line")
    sender: str = Field(..., description="Sender email address")
    sender_domain: str = Field(..., description="Domain portion of sender address")
    body: str = Field(..., description="Full email body text")
    timestamp: str = Field(..., description="ISO-8601 timestamp of receipt")
    thread_length: int = Field(..., description="Number of messages in this thread")
    has_attachments: bool = Field(..., description="Whether attachments are present")
    inbox_position: int = Field(..., description="1-indexed position in inbox (current email number)")
    total_emails: int = Field(..., description="Total emails in this episode")
    step_count: int = Field(..., description="Number of steps taken so far this episode")
    done: bool = Field(False, description="True when all emails have been triaged")
    task_id: str = Field(..., description="Active task identifier")


# ---------------------------------------------------------------------------
# Action — what the agent decides
# ---------------------------------------------------------------------------
class TriageAction(BaseModel):
    """Agent's triage decision for the current email."""
    priority: str = Field(
        ...,
        description="Priority: urgent | high | normal | low | spam",
    )
    category: str = Field(
        ...,
        description=(
            "Category: support | billing | sales | internal | legal | hr | "
            "it | complaint | inquiry | spam | other"
        ),
    )
    action: str = Field(
        ...,
        description="Action: reply | forward | archive | delete | escalate | flag | snooze",
    )
    reply_draft: Optional[str] = Field(
        None,
        description="Draft reply text. Required when action='reply', optional otherwise.",
    )
    forward_to: Optional[str] = Field(
        None,
        description="Destination email/team for forward/escalate actions.",
    )
    tags: Optional[List[str]] = Field(
        default_factory=list,
        description="Optional free-form tags to apply to the email.",
    )


# ---------------------------------------------------------------------------
# Reward — per-step signal
# ---------------------------------------------------------------------------
class StepReward(BaseModel):
    """Reward signal returned after each step."""
    reward: float = Field(..., description="Step reward in range [0.0, 1.0]")
    priority_score: float = Field(..., description="Score for priority classification (0.0-1.0)")
    category_score: float = Field(..., description="Score for category classification (0.0-1.0)")
    action_score: float = Field(..., description="Score for action choice (0.0-1.0)")
    reply_score: Optional[float] = Field(None, description="Score for reply quality if applicable (0.0-1.0)")
    feedback: str = Field(..., description="Human-readable grader feedback")
    penalties: List[str] = Field(default_factory=list, description="List of applied penalties")


# ---------------------------------------------------------------------------
# Step response — full return from /step
# ---------------------------------------------------------------------------
class StepResponse(BaseModel):
    """Full response returned by the step() endpoint."""
    observation: Optional[EmailObservation] = Field(
        None, description="Next email observation (None if done)"
    )
    reward: StepReward
    done: bool = Field(..., description="True when episode is complete")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional info dict")


# ---------------------------------------------------------------------------
# Episode state — full internal state snapshot
# ---------------------------------------------------------------------------
class EpisodeState(BaseModel):
    """Full state of the current episode, returned by /state."""
    task_id: str
    step_count: int
    email_index: int
    total_emails: int
    done: bool
    cumulative_reward: float
    per_step_rewards: List[float]
    emails_processed: List[str]
    current_email_id: Optional[str]


# ---------------------------------------------------------------------------
# Reset response
# ---------------------------------------------------------------------------
class ResetResponse(BaseModel):
    """Response from reset() — initial observation."""
    observation: EmailObservation
    task_id: str
    total_emails: int
    message: str = "Episode reset. Good luck!"


# ---------------------------------------------------------------------------
# Task info
# ---------------------------------------------------------------------------
class TaskInfo(BaseModel):
    id: str
    name: str
    difficulty: str
    description: str
    max_steps: int
    email_count: int


class TaskListResponse(BaseModel):
    tasks: List[TaskInfo]