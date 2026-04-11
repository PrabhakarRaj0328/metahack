"""
inference.py — Email Triage OpenEnv Baseline Inference Script

Runs a baseline LLM agent against all 3 tasks and emits structured
[START] / [STEP] / [END] logs to stdout for automated evaluation.

Usage:
    python inference.py

Environment variables:
    API_BASE_URL   — LLM API base URL (OpenAI-compatible)
    MODEL_NAME     — Model identifier
    HF_TOKEN       — API key (used as the OpenAI client API key)
    ENV_BASE_URL   — Email Triage env URL (default: http://localhost:8000)
"""

from __future__ import annotations
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

TASKS = ["task_easy", "task_medium", "task_hard"]
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# OpenAI client (OpenAI-compatible)
# ---------------------------------------------------------------------------
client = OpenAI(api_key=HF_TOKEN or "sk-placeholder", base_url=API_BASE_URL)


# ---------------------------------------------------------------------------
# Env API helpers
# ---------------------------------------------------------------------------
def env_reset(task_id: str) -> Dict[str, Any]:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


def env_state() -> Dict[str, Any]:
    r = requests.get(f"{ENV_BASE_URL}/state", timeout=10)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Agent prompt builder
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert email triage assistant. For each email you receive,
you must output a JSON object with EXACTLY these fields:

{
  "priority": "<urgent|high|normal|low|spam>",
  "category": "<support|billing|sales|internal|legal|hr|it|complaint|inquiry|spam|other>",
  "action": "<reply|forward|archive|delete|escalate|flag|snooze>",
  "reply_draft": "<string or null>",
  "forward_to": "<email or null>",
  "tags": []
}

Guidelines:
- priority: urgent=needs action NOW (outages, legal threats, security). high=today. normal=this week. low=FYI. spam=junk/phishing.
- category: pick the best-fit department or topic.
- action: reply=you draft a response. escalate=needs a senior person/legal/HR immediately. forward=send to another team. archive=save but no action. delete=spam/junk.
- reply_draft: REQUIRED when action=reply. Write a professional, concise reply addressing the sender's needs. Null otherwise.
- forward_to: include a team/person when action=forward or escalate (e.g. "legal@company.com", "IT Help Desk").
- NEVER reply to spam or phishing emails.
- Legal threats, data breaches, harassment complaints, and media inquiries must ALWAYS be escalated.

Output ONLY valid JSON. No prose, no markdown fences."""


def build_user_prompt(obs: Dict[str, Any]) -> str:
    return f"""Email to triage:
Subject: {obs['subject']}
From: {obs['sender']}
Received: {obs['timestamp']}
Thread length: {obs['thread_length']}
Has attachments: {obs['has_attachments']}

Body:
{obs['body']}

Progress: email {obs['inbox_position']} of {obs['total_emails']} | step {obs['step_count']}

Respond with JSON only."""


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------
def call_llm(user_prompt: str, retries: int = MAX_RETRIES) -> Dict[str, Any]:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=600,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt == retries - 1:
                # Return a safe fallback action
                return {
                    "priority": "normal",
                    "category": "other",
                    "action": "archive",
                    "reply_draft": None,
                    "forward_to": None,
                    "tags": [],
                }
            time.sleep(1)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------
def run_task(task_id: str) -> Dict[str, Any]:
    reset_data = env_reset(task_id)
    obs = reset_data["observation"]

    step_results = []
    episode_done = False
    step_num = 0

    while not episode_done:
        step_num += 1
        user_prompt = build_user_prompt(obs)

        t0 = time.time()
        action = call_llm(user_prompt)
        latency = round(time.time() - t0, 3)

        # Ensure required fields present
        action.setdefault("priority", "normal")
        action.setdefault("category", "other")
        action.setdefault("action", "archive")
        action.setdefault("reply_draft", None)
        action.setdefault("forward_to", None)
        action.setdefault("tags", [])

        step_data = env_step(action)

        reward_info = step_data["reward"]
        step_reward = reward_info["reward"]
        episode_done = step_data["done"]

        step_record = {
            "task_id": task_id,
            "step": step_num,
            "email_id": obs.get("email_id"),
            "action": action,
            "reward": step_reward,
            "priority_score": reward_info.get("priority_score"),
            "category_score": reward_info.get("category_score"),
            "action_score": reward_info.get("action_score"),
            "reply_score": reward_info.get("reply_score"),
            "feedback": reward_info.get("feedback"),
            "penalties": reward_info.get("penalties", []),
            "done": episode_done,
            "latency_s": latency,
        }
        step_results.append(step_record)

        # Emit [STEP] log
        print(json.dumps({"type": "STEP", **step_record}), flush=True)

        if not episode_done:
            obs = step_data["observation"]

    # Final state
    final_state = env_state()
    total_reward = final_state["cumulative_reward"]
    num_emails = final_state["total_emails"]
    mean_reward = round(total_reward / max(num_emails, 1), 4)

    return {
        "task_id": task_id,
        "total_steps": step_num,
        "cumulative_reward": total_reward,
        "mean_reward_per_email": mean_reward,
        "emails_processed": final_state["emails_processed"],
        "step_results": step_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()

    # Verify env is up
    try:
        r = requests.get(f"{ENV_BASE_URL}/health", timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(json.dumps({"type": "ERROR", "message": f"Environment not reachable: {e}"}))
        sys.exit(1)

    all_results = {}

    # Emit [START]
    print(json.dumps({
        "type": "START",
        "model": MODEL_NAME,
        "api_base": API_BASE_URL,
        "env_url": ENV_BASE_URL,
        "tasks": TASKS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }), flush=True)

    for task_id in TASKS:
        print(json.dumps({"type": "STEP", "task_id": task_id, "step": 0, "event": "task_start"}), flush=True)
        task_result = run_task(task_id)
        all_results[task_id] = {
            "score": task_result["mean_reward_per_email"],
            "cumulative_reward": task_result["cumulative_reward"],
            "total_steps": task_result["total_steps"],
        }
        print(json.dumps({
            "type": "STEP",
            "task_id": task_id,
            "step": task_result["total_steps"],
            "event": "task_end",
            "score": task_result["mean_reward_per_email"],
        }), flush=True)

    elapsed = round(time.time() - start_time, 2)
    overall_mean = round(
        sum(v["score"] for v in all_results.values()) / len(all_results), 4
    )

    # Emit [END]
    print(json.dumps({
        "type": "END",
        "model": MODEL_NAME,
        "elapsed_seconds": elapsed,
        "task_scores": {tid: v["score"] for tid, v in all_results.items()},
        "overall_mean_score": overall_mean,
        "details": all_results,
    }), flush=True)


if __name__ == "__main__":
    main()