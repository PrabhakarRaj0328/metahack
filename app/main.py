"""
Email Triage OpenEnv — FastAPI server
Implements the OpenEnv spec: reset(), step(), state() + /tasks + /health
"""

from __future__ import annotations
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.models import (
    EmailObservation,
    TriageAction,
    StepResponse,
    EpisodeState,
    ResetResponse,
    TaskInfo,
    TaskListResponse,
)
from app.episode import EpisodeManager
from data.emails import TASK_EMAILS

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Email Triage OpenEnv",
    description=(
        "Real-world email triage environment. An AI agent reads incoming emails "
        "and must classify priority, category, choose an action, and draft replies. "
        "Implements the full OpenEnv spec."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global episode manager (one episode at a time, protected by a lock)
_episode = EpisodeManager()
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Request/Response schemas for endpoints
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    task_id: str = "task_easy"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "email-triage-openenv", "version": "1.0.0"}


@app.post("/reset", response_model=ResetResponse)
def reset(request: Optional[ResetRequest] = None):
    """
    Reset the environment and return the first email observation.
    task_id must be one of: task_easy, task_medium, task_hard.
    Body is optional — omitting it defaults to task_easy.
    """
    # Accept POST /reset with no body, empty body, or {"task_id": "..."}
    task_id = (request.task_id if request is not None else None) or "task_easy"

    with _lock:
        try:
            obs = _episode.reset(task_id=task_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return ResetResponse(
        observation=obs,
        task_id=task_id,
        total_emails=len(TASK_EMAILS[task_id]),
        message=f"Episode reset for task '{task_id}'. Triage {len(TASK_EMAILS[task_id])} emails.",
    )


@app.post("/step", response_model=StepResponse)
def step(action: TriageAction):
    """
    Submit a triage action for the current email.
    Returns: next observation, reward, done flag, and info dict.
    """
    with _lock:
        if not _episode.is_initialized:
            raise HTTPException(status_code=400, detail="Call /reset first.")
        try:
            response = _episode.step(action)
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return response


@app.get("/state", response_model=EpisodeState)
def state():
    """Return the current episode state (step count, rewards, progress)."""
    with _lock:
        if not _episode.is_initialized:
            raise HTTPException(status_code=400, detail="Call /reset first.")
        return _episode.state()


@app.get("/tasks", response_model=TaskListResponse)
def list_tasks():
    """List all available tasks with metadata."""
    tasks = [
        TaskInfo(
            id="task_easy",
            name="Priority Classification",
            difficulty="easy",
            description=(
                "Classify the priority (urgent/high/normal/low/spam) of 10 emails. "
                "Reward: exact=1.0, adjacent=0.5, 2-away=0.2."
            ),
            max_steps=15,
            email_count=len(TASK_EMAILS["task_easy"]),
        ),
        TaskInfo(
            id="task_medium",
            name="Triage and Routing",
            difficulty="medium",
            description=(
                "Classify priority + category + action for 10 emails. "
                "Weighted reward: priority 40%, category 30%, action 30%."
            ),
            max_steps=15,
            email_count=len(TASK_EMAILS["task_medium"]),
        ),
        TaskInfo(
            id="task_hard",
            name="Full Inbox Management",
            difficulty="hard",
            description=(
                "Full triage + reply drafting for 15 emails including tricky cases "
                "(phishing, legal, HR-sensitive). Reward includes reply quality scoring."
            ),
            max_steps=20,
            email_count=len(TASK_EMAILS["task_hard"]),
        ),
    ]
    return TaskListResponse(tasks=tasks)


@app.get("/")
def root():
    """Root endpoint with usage info."""
    return {
        "name": "Email Triage OpenEnv",
        "version": "1.0.0",
        "endpoints": {
            "POST /reset":  "Reset episode. Body: {task_id: str}",
            "POST /step":   "Submit action. Body: TriageAction",
            "GET  /state":  "Get current episode state",
            "GET  /tasks":  "List available tasks",
            "GET  /health": "Health check",
            "GET  /docs":   "Interactive API docs (Swagger)",
        },
        "tasks": ["task_easy", "task_medium", "task_hard"],
    }


def start():
    """Entry point for [project.scripts] server = 'app.main:start'"""
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, workers=1)