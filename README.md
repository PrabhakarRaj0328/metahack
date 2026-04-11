---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - email
  - triage
  - nlp
  - real-world
  - agent-evaluation
license: apache-2.0
---

# 📧 Email Triage OpenEnv

A **real-world OpenEnv environment** where an AI agent manages an inbox — classifying email priority, routing to the right team, choosing actions, and drafting professional replies.

This environment models a task performed millions of times daily by knowledge workers, executive assistants, and customer support agents. It provides rich partial-credit reward signals that make it useful for both training and evaluating LLM-based agents.

---

## Environment Description & Motivation

Email triage is a genuine, high-value knowledge-work task with:

- **Clear ground truth** — priority, category, and action can be objectively graded
- **Natural difficulty gradient** — from simple spam detection to nuanced legal/HR escalations
- **Rich partial credit** — adjacent priority mistakes, acceptable-alternative actions, and reply quality all receive partial scores
- **Real failure modes** — replying to phishing, deleting legal emails, ignoring urgent security alerts

This fills a gap in the current OpenEnv ecosystem: a language-heavy, multi-label classification + generation task grounded in real workplace behaviour.

---

## Observation Space

Each step the agent receives an `EmailObservation`:

| Field | Type | Description |
|---|---|---|
| `email_id` | string | Unique email identifier |
| `subject` | string | Email subject line |
| `sender` | string | Full sender address |
| `sender_domain` | string | Domain portion of sender |
| `body` | string | Full email body text |
| `timestamp` | string | ISO-8601 receipt timestamp |
| `thread_length` | int | Number of messages in the thread |
| `has_attachments` | bool | Whether attachments are present |
| `inbox_position` | int | 1-indexed current email number |
| `total_emails` | int | Total emails in this episode |
| `step_count` | int | Steps taken so far |
| `done` | bool | True when all emails processed |
| `task_id` | string | Active task identifier |

---

## Action Space

The agent submits a `TriageAction`:

| Field | Type | Values / Description |
|---|---|---|
| `priority` | string | `urgent` \| `high` \| `normal` \| `low` \| `spam` |
| `category` | string | `support` \| `billing` \| `sales` \| `internal` \| `legal` \| `hr` \| `it` \| `complaint` \| `inquiry` \| `spam` \| `other` |
| `action` | string | `reply` \| `forward` \| `archive` \| `delete` \| `escalate` \| `flag` \| `snooze` |
| `reply_draft` | string? | Draft reply text. Required when `action=reply` |
| `forward_to` | string? | Destination for forward/escalate actions |
| `tags` | string[] | Optional free-form tags |

---

## Tasks

### Task 1 — Priority Classification (Easy)
**10 emails** — classify priority only (urgent / high / normal / low / spam).

- Reward: `1.0` exact, `0.5` one level off, `0.2` two levels off
- Penalty: `-0.3` for replying to spam
- Expected baseline score: **~0.72**

### Task 2 — Triage and Routing (Medium)
**10 emails** — classify priority + category + choose correct action.

- Reward: `40% priority + 30% category + 30% action`
- Partial credit for related categories (e.g. `legal` ↔ `complaint`)
- Penalty: `-0.5` for deleting legal emails, `-0.3` for replying to spam
- Expected baseline score: **~0.58**

### Task 3 — Full Inbox Management (Hard)
**15 emails** including tricky cases: phishing detection, legal/HR sensitivity, media inquiries, data breaches.

- Reward: `25% priority + 20% category + 25% action + 30% reply quality`
- Reply quality scored by keyword coverage + length heuristic
- Phishing detection bonus: correctly spam+delete a phishing email = full score
- Additional penalties for drafting replies to escalation-required emails
- Expected baseline score: **~0.41**

---

## Reward Function Design

### Partial credit philosophy
The reward function is designed to provide **signal throughout the trajectory**, not just at the end:

```
Priority distance scoring:
  spam → low → normal → high → urgent
  exact:    1.0
  ±1 step:  0.5
  ±2 steps: 0.2
  ±3+ steps: 0.0

Category group partial credit (0.4):
  {legal, complaint}
  {support, inquiry}
  {billing, sales}
  {hr, internal}

Action acceptability:
  Exact match:     1.0
  Acceptable alt:  0.6  (escalate ↔ forward, archive ↔ snooze, etc.)
  Neutral:         0.2
  Destructive:     0.0  (delete/archive on urgent/high email)

Penalties (each -0.1, cumulative):
  - Replying to spam/phishing
  - Deleting a legal email
  - No reply when reply_required=True
  - Drafting a reply for an escalation email
```

---

## API Reference

### `POST /reset`
```json
{ "task_id": "task_easy" }
```
Returns the first `EmailObservation` and episode metadata.

### `POST /step`
```json
{
  "priority": "urgent",
  "category": "it",
  "action": "escalate",
  "reply_draft": null,
  "forward_to": "security@company.com",
  "tags": ["security", "breach"]
}
```
Returns `{ observation, reward, done, info }`.

### `GET /state`
Returns current episode state: step count, cumulative reward, emails processed.

### `GET /tasks`
Lists all available tasks with metadata.

### `GET /health`
Health check — returns `{"status": "ok"}`.

---

## Setup & Usage

### Docker (recommended)

```bash
# Build
docker build -t email-triage-env .

# Run the environment server
docker run -p 8000:8000 email-triage-env

# In another terminal, run baseline inference
docker run --network host \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="sk-..." \
  -e ENV_BASE_URL="http://localhost:8000" \
  email-triage-env python inference.py
```

### Local development

```bash
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload --port 8000

# Run baseline
API_BASE_URL="https://api.openai.com/v1" \
MODEL_NAME="gpt-4o-mini" \
HF_TOKEN="sk-..." \
python inference.py
```

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | LLM API base URL (OpenAI-compatible) |
| `MODEL_NAME` | Yes | Model identifier |
| `HF_TOKEN` | Yes | API key for the LLM |
| `ENV_BASE_URL` | No | Environment URL (default: `http://localhost:8000`) |

---

## Baseline Scores

Measured with `gpt-4o-mini` at temperature 0:

| Task | Score | Notes |
|---|---|---|
| task_easy | 0.72 | Strong on obvious cases, misses nuanced priority |
| task_medium | 0.58 | Routing errors on ambiguous cases |
| task_hard | 0.41 | Reply quality and phishing detection drag score |
| **Overall** | **0.57** | |

---

## Stdout Log Format

The inference script emits structured JSON logs:

```
{"type": "START", "model": "...", "tasks": [...], ...}
{"type": "STEP", "task_id": "task_easy", "step": 1, "email_id": "e001", "reward": 1.0, ...}
...
{"type": "END", "task_scores": {"task_easy": 0.72, ...}, "overall_mean_score": 0.57, ...}
```

---

## OpenEnv Spec Compliance

- ✅ `openenv.yaml` with full metadata
- ✅ Typed Pydantic models: `EmailObservation`, `TriageAction`, `StepReward`
- ✅ `POST /reset` → returns initial observation
- ✅ `POST /step` → returns observation, reward, done, info
- ✅ `GET /state` → returns current state
- ✅ 3 tasks with graders (easy → medium → hard)
- ✅ Rewards in [0.0, 1.0] with partial progress signals
- ✅ Deterministic graders (no randomness in scoring)
- ✅ Working Dockerfile
- ✅ `inference.py` with `[START]`/`[STEP]`/`[END]` structured logs
- ✅ OpenAI client for all LLM calls
- ✅ `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from environment variables

---

## Project Structure

```
email-triage-env/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI server (reset/step/state endpoints)
│   ├── models.py        # Pydantic typed models
│   ├── episode.py       # Episode state manager
│   └── graders.py       # Task graders with partial credit
├── data/
│   ├── __init__.py
│   └── emails.py        # Email dataset with gold labels
├── inference.py         # Baseline inference script
├── openenv.yaml         # OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## License

Apache 2.0