"""
Episode manager for the Email Triage environment.
Handles reset(), step(), and state() lifecycle.
"""

from __future__ import annotations
import copy
from typing import Dict, Any, List, Optional

from app.models import (
    EmailObservation,
    TriageAction,
    StepResponse,
    EpisodeState,
    StepReward,
)
from app.graders import grade
from data.emails import TASK_EMAILS

MAX_STEPS_PER_TASK = {
    "task_easy":   15,
    "task_medium": 15,
    "task_hard":   20,
}


def _make_observation(email: Dict[str, Any], idx: int, total: int, step: int, task_id: str, done: bool = False) -> EmailObservation:
    return EmailObservation(
        email_id=email["email_id"],
        subject=email["subject"],
        sender=email["sender"],
        sender_domain=email["sender_domain"],
        body=email["body"],
        timestamp=email["timestamp"],
        thread_length=email["thread_length"],
        has_attachments=email["has_attachments"],
        inbox_position=idx + 1,
        total_emails=total,
        step_count=step,
        done=done,
        task_id=task_id,
    )


class EpisodeManager:
    """
    Manages a single episode.  One episode = one task = one inbox of emails.
    Thread-safe per-request if a new instance is used per session,
    or a single shared instance guarded by a lock.
    """

    def __init__(self) -> None:
        self._task_id: Optional[str] = None
        self._emails: List[Dict[str, Any]] = []
        self._email_index: int = 0
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0
        self._per_step_rewards: List[float] = []
        self._emails_processed: List[str] = []
        self._done: bool = False

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------
    def reset(self, task_id: str = "task_easy") -> EmailObservation:
        if task_id not in TASK_EMAILS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASK_EMAILS)}")

        self._task_id = task_id
        self._emails = copy.deepcopy(TASK_EMAILS[task_id])
        self._email_index = 0
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._per_step_rewards = []
        self._emails_processed = []
        self._done = False

        return _make_observation(
            self._emails[0],
            idx=0,
            total=len(self._emails),
            step=0,
            task_id=self._task_id,
        )

    # ------------------------------------------------------------------
    # step(action)
    # ------------------------------------------------------------------
    def step(self, action: TriageAction) -> StepResponse:
        if self._task_id is None:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        max_steps = MAX_STEPS_PER_TASK[self._task_id]
        current_email = self._emails[self._email_index]

        # Grade the action
        reward: StepReward = grade(self._task_id, action, current_email)

        # Update state
        self._step_count += 1
        self._cumulative_reward += reward.reward
        self._per_step_rewards.append(reward.reward)
        self._emails_processed.append(current_email["email_id"])
        self._email_index += 1

        # Determine done conditions
        all_emails_done = self._email_index >= len(self._emails)
        step_limit_hit = self._step_count >= max_steps
        self._done = all_emails_done or step_limit_hit

        # Build next observation
        next_obs: Optional[EmailObservation] = None
        if not self._done:
            next_obs = _make_observation(
                self._emails[self._email_index],
                idx=self._email_index,
                total=len(self._emails),
                step=self._step_count,
                task_id=self._task_id,
            )
        else:
            # Final done observation (placeholder)
            next_obs = EmailObservation(
                email_id="DONE",
                subject="",
                sender="",
                sender_domain="",
                body="",
                timestamp="",
                thread_length=0,
                has_attachments=False,
                inbox_position=len(self._emails),
                total_emails=len(self._emails),
                step_count=self._step_count,
                done=True,
                task_id=self._task_id,
            )

        info = {
            "email_id": current_email["email_id"],
            "step": self._step_count,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "emails_remaining": len(self._emails) - self._email_index,
            "step_limit": max_steps,
        }

        return StepResponse(
            observation=next_obs,
            reward=reward,
            done=self._done,
            info=info,
        )

    # ------------------------------------------------------------------
    # state()
    # ------------------------------------------------------------------
    def state(self) -> EpisodeState:
        current_email_id = None
        if not self._done and self._email_index < len(self._emails):
            current_email_id = self._emails[self._email_index]["email_id"]

        return EpisodeState(
            task_id=self._task_id or "none",
            step_count=self._step_count,
            email_index=self._email_index,
            total_emails=len(self._emails),
            done=self._done,
            cumulative_reward=round(self._cumulative_reward, 4),
            per_step_rewards=self._per_step_rewards,
            emails_processed=self._emails_processed,
            current_email_id=current_email_id,
        )

    @property
    def is_initialized(self) -> bool:
        return self._task_id is not None