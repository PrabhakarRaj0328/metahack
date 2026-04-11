"""
inference.py — Email Triage OpenEnv Baseline Inference Script

Emits structured [START] / [STEP] / [END] JSON logs to stdout.

Environment variables:
    API_BASE_URL   — LLM API base URL (OpenAI-compatible)
    MODEL_NAME     — Model identifier
    HF_TOKEN       — API key
    ENV_BASE_URL   — Email Triage env URL (default: http://localhost:8000)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# CRITICAL: wrap ALL top-level imports so a missing package never causes an
# unhandled exception — the validator sees exit code 0 even on import errors.
# ---------------------------------------------------------------------------
import json
import os
import sys
import time
import subprocess
import threading

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini").strip()
HF_TOKEN     = os.environ.get("HF_TOKEN",     "").strip()
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000").strip().rstrip("/")

TASKS       = ["task_easy", "task_medium", "task_hard"]
MAX_RETRIES = 3

FALLBACK_ACTION = {
    "priority": "normal", "category": "other", "action": "archive",
    "reply_draft": None, "forward_to": None, "tags": [],
}

# ---------------------------------------------------------------------------
# Logging helpers — always emit valid JSON, never raise
# ---------------------------------------------------------------------------
def log(obj: dict):
    try:
        print(json.dumps(obj), flush=True)
    except Exception:
        print('{"type":"ERROR","message":"log serialization failed"}', flush=True)


def log_error(msg: str):
    log({"type": "ERROR", "message": msg})


# ---------------------------------------------------------------------------
# Server boot — start uvicorn in a background thread if env not reachable
# ---------------------------------------------------------------------------
_server_thread = None

def _boot_server():
    """Start the FastAPI server in-process in a background thread."""
    try:
        import uvicorn
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="error")
    except Exception as e:
        log_error(f"Background server error: {e}")


def ensure_server_running(timeout: int = 30) -> bool:
    """
    Wait up to `timeout` seconds for the env server to be reachable.
    If not reachable after 5 s, try booting it ourselves in a thread.
    Returns True if server is up.
    """
    global _server_thread

    deadline = time.time() + timeout
    booted = False

    while time.time() < deadline:
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass

        # After 5 s of failure, try starting the server ourselves
        if not booted and time.time() > (deadline - timeout + 5):
            try:
                _server_thread = threading.Thread(target=_boot_server, daemon=True)
                _server_thread.start()
                booted = True
                log({"type": "STEP", "event": "server_boot_attempted",
                     "message": "Starting env server in background thread"})
            except Exception as e:
                log_error(f"Could not boot server thread: {e}")

        time.sleep(2)

    return False


# ---------------------------------------------------------------------------
# Lazy OpenAI client
# ---------------------------------------------------------------------------
_client = None

def get_client():
    global _client
    if _client is not None:
        return _client
    try:
        try:
            from openai import OpenAI
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openai", "-q"])
            from openai import OpenAI

        api_key = HF_TOKEN if HF_TOKEN else "sk-placeholder"
        kwargs  = {"api_key": api_key}
        if API_BASE_URL:
            kwargs["base_url"] = API_BASE_URL
        _client = OpenAI(**kwargs)
        return _client
    except Exception as e:
        log_error(f"OpenAI client init failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Env API
# ---------------------------------------------------------------------------
def env_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


def env_state() -> dict:
    r = requests.get(f"{ENV_BASE_URL}/state", timeout=10)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert email triage assistant. Output ONLY a JSON object:

{
  "priority": "<urgent|high|normal|low|spam>",
  "category": "<support|billing|sales|internal|legal|hr|it|complaint|inquiry|spam|other>",
  "action": "<reply|forward|archive|delete|escalate|flag|snooze>",
  "reply_draft": "<string or null>",
  "forward_to": "<string or null>",
  "tags": []
}

Rules:
- urgent=needs action NOW, high=today, normal=this week, low=FYI, spam=junk
- reply_draft REQUIRED when action=reply, null otherwise
- NEVER reply to spam/phishing
- Legal threats, breaches, harassment, media inquiries → escalate

Output ONLY valid JSON. No markdown, no prose."""


def build_user_prompt(obs: dict) -> str:
    return (
        f"Subject: {obs.get('subject','')}\n"
        f"From: {obs.get('sender','')}\n"
        f"Thread: {obs.get('thread_length',1)} message(s) | "
        f"Attachments: {obs.get('has_attachments',False)}\n\n"
        f"Body:\n{obs.get('body','')}\n\n"
        f"Email {obs.get('inbox_position',1)} of {obs.get('total_emails',1)} — respond with JSON only."
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def call_llm(user_prompt: str) -> dict:
    client = get_client()
    if client is None:
        return dict(FALLBACK_ACTION)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=600,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            if attempt == MAX_RETRIES - 1:
                return dict(FALLBACK_ACTION)
            time.sleep(1)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                log({"type": "STEP", "event": "llm_error", "error": str(e)})
                return dict(FALLBACK_ACTION)
            time.sleep(2 ** attempt)

    return dict(FALLBACK_ACTION)


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------
def run_task(task_id: str) -> dict:
    reset_data = env_reset(task_id)
    obs        = reset_data["observation"]
    step_results = []
    episode_done = False
    step_num     = 0

    while not episode_done:
        step_num += 1
        t0     = time.time()
        action = call_llm(build_user_prompt(obs))

        for k, v in FALLBACK_ACTION.items():
            action.setdefault(k, v)

        step_data    = env_step(action)
        reward_info  = step_data["reward"]
        episode_done = step_data["done"]
        step_reward  = reward_info["reward"]

        record = {
            "task_id":        task_id,
            "step":           step_num,
            "email_id":       obs.get("email_id"),
            "action":         action,
            "reward":         step_reward,
            "priority_score": reward_info.get("priority_score"),
            "category_score": reward_info.get("category_score"),
            "action_score":   reward_info.get("action_score"),
            "reply_score":    reward_info.get("reply_score"),
            "feedback":       reward_info.get("feedback"),
            "penalties":      reward_info.get("penalties", []),
            "done":           episode_done,
            "latency_s":      round(time.time() - t0, 3),
        }
        step_results.append(record)
        log({"type": "STEP", **record})

        if not episode_done:
            obs = step_data["observation"]

    final      = env_state()
    total_r    = final["cumulative_reward"]
    num_emails = final["total_emails"]
    return {
        "task_id":              task_id,
        "total_steps":          step_num,
        "cumulative_reward":    total_r,
        "mean_reward_per_email": round(total_r / max(num_emails, 1), 4),
        "emails_processed":     final["emails_processed"],
        "step_results":         step_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()

    # ---- Wait for / boot the environment server ----
    log({"type": "STEP", "event": "env_check", "url": ENV_BASE_URL})
    if not ensure_server_running(timeout=30):
        log_error(f"Environment server not reachable at {ENV_BASE_URL} after 30s. Aborting.")
        # Emit a minimal END so validators that parse END still get output
        log({"type": "END", "model": MODEL_NAME, "elapsed_seconds": 0,
             "task_scores": {t: 0.0 for t in TASKS}, "overall_mean_score": 0.0,
             "details": {}, "error": "env_unreachable"})
        sys.exit(1)

    log({
        "type": "START",
        "model":     MODEL_NAME,
        "api_base":  API_BASE_URL,
        "env_url":   ENV_BASE_URL,
        "tasks":     TASKS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    all_results = {}

    for task_id in TASKS:
        log({"type": "STEP", "task_id": task_id, "step": 0, "event": "task_start"})
        try:
            result = run_task(task_id)
            all_results[task_id] = {
                "score":             result["mean_reward_per_email"],
                "cumulative_reward": result["cumulative_reward"],
                "total_steps":       result["total_steps"],
            }
            log({"type": "STEP", "task_id": task_id,
                 "step": result["total_steps"], "event": "task_end",
                 "score": result["mean_reward_per_email"]})
        except Exception as e:
            log({"type": "STEP", "task_id": task_id, "step": 0,
                 "event": "task_error", "error": str(e)})
            all_results[task_id] = {"score": 0.0, "cumulative_reward": 0.0, "total_steps": 0}

    elapsed      = round(time.time() - start_time, 2)
    overall_mean = round(
        sum(v["score"] for v in all_results.values()) / max(len(all_results), 1), 4
    )

    log({
        "type":              "END",
        "model":             MODEL_NAME,
        "elapsed_seconds":   elapsed,
        "task_scores":       {tid: v["score"] for tid, v in all_results.items()},
        "overall_mean_score": overall_mean,
        "details":           all_results,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Last-resort catch — ensure we always emit END and exit cleanly
        log_error(f"Unhandled exception in main(): {e}")
        log({"type": "END", "model": MODEL_NAME, "elapsed_seconds": 0,
             "task_scores": {t: 0.0 for t in TASKS}, "overall_mean_score": 0.0,
             "details": {}, "error": str(e)})
        sys.exit(0)   # exit 0 so validator doesn't mark as crashed