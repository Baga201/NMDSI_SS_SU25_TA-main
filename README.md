# Multi-Agent Rehabilitation Research Platform

Config-driven multi-agent system for rehabilitation research, built on Azure AI Projects/Agents. Local-first, with a FastAPI backend and a built-in HTML UI (Jinja2 + HTMX) for a transparent group-chat experience with explicit sender@receiver labeling, agent delegation, retrieval-augmented responses, and embedded run traces.

Key goals: accuracy, transparency, and reproducibility for multi-agent experiments in a research setting.

## Features

- Multi-agent orchestration using Azure AI Agents (leader + connected workers)
- Retrieval-augmented generation via FileSearchTool (vector store-backed)
- Deterministic runs (create + poll; avoids flaky streaming dependencies)
- FastAPI backend with built-in HTML UI (Jinja2 + HTMX)
	- WhatsApp-style group chat with sender @ receiver
	- Per-message structured output view and validation
	- Inline trace (tool input/output) inside each bubble
	- Live run events panel with typing indicators
	- Role adapter (final answer adapted for Student/Executive/Engineer/etc.)
	- Safety + heuristic Critic signals
	- Diagnostics panel for leader↔worker wiring and env ID mismatches
- Live updates via SSE (/events_sse) with JSON polling fallback (/events)
- One-click “Reset agents” (only deletes agents created by this session)
- Optional OpenTelemetry → Azure Monitor integration
- Configurable via YAML and environment variables

## Architecture (at a glance)

- FastAPI backend: `scripts/backend.py`
	- Session/thread management, agent creation (leader + workers), runs, streaming, trace serialization, connected-agent output surfacing
- CLI runner: `scripts/run_team.py`
	- Local sanity checks: create agents, run a query, print messages and citations
- Team config: `configs/team.yaml`
	- Defines leader/workers, models, instructions, and tools (e.g., file_search)
- Data ingest examples: `scripts/ingest_data.py`
- Optional Azure infra: `infra/` (Bicep templates)

## Prerequisites

- Python 3.10–3.13
- Azure login locally (Azure CLI or VS Code Azure sign-in) so `DefaultAzureCredential` works
- An Azure AI Foundry project endpoint with model deployments and (optionally) a vector store

## Quickstart (Windows, PowerShell)

1) Clone

```powershell
git clone <repo-url>
cd NMDSI_SS_SU25_TA
```

2) Install dependencies

Option A — Using your system Python (works well on Windows):

```powershell
C:/Python313/python.exe -m ensurepip --upgrade
C:/Python313/python.exe -m pip install -r requirements.txt
```

Option B — Using uv (fast virtualenv):

```powershell
iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

3) Set environment variables

```powershell
$env:PROJECT_ENDPOINT = "https://<your-project>.<region>.inference.ai.azure.com"
# Required if librarian_rag is enabled in team.yaml
$env:VECTOR_STORE_ID = "<your-vector-store-id>"
# Optional: read team config from a custom path (defaults to configs/team.yaml)
$env:TEAM_CONFIG_PATH = "configs/team.yaml"
# (no frontend URL needed; use the built-in HTML UI below)
# Optional: Azure Monitor (enables tracing)
$env:APPLICATIONINSIGHTS_CONNECTION_STRING = "InstrumentationKey=..."

# Optional: Reuse pre-provisioned agents by ID (skips creating/deleting)
# Leader
$env:LEADER_AGENT_ID = "asst_123456789"
# Workers — use AGENT_ID_{WORKER_NAME} uppercased (examples below)
$env:AGENT_ID_LIBRARIAN_RAG = "asst_lib_abcdef"
$env:AGENT_ID_THERAPY_EXPERT = "asst_th_abcdef"
$env:AGENT_ID_ROBOTICS_EXPERT = "asst_ro_abcdef"
```

4) Start the backend

```powershell
# System python
C:/Python313/python.exe -m uvicorn scripts.backend:app --host 0.0.0.0 --port 8000 --reload

# Or uv-managed venv
uv run uvicorn scripts.backend:app --host 0.0.0.0 --port 8000 --reload
```

The backend exposes:
- GET `/` → status and endpoint list
- POST `/session` → create a session (one thread per session)
- POST `/chat` → send message (and optional image), run the leader, return chat history + run trace
- GET `/events_sse` → Server-Sent Events (primary live event stream for the UI)
- GET `/events` → JSON polling fallback for run events
- GET `/ui` → messenger-like HTML UI (Jinja2 + HTMX)
- GET `/ui/diagnostics` → wiring/ID checks for leader and workers
- POST `/reset_agents` → delete leader/workers for a session; optionally start a new thread

5) Open the HTML UI

Visit:

```
http://localhost:8000/ui
```

What you’ll see:
- Chat pane with agent bubbles and structured output
- “Trace for this agent” expander showing tool input/output from the latest run
- Live run events on the right
- Sidebar “Reset agents” to cleanly rebuild the team (optionally with a new thread)

## CLI sanity runner

You can exercise the flow without the UI:

```powershell
uv run python .\scripts\run_team.py --question "Outline early post-stroke gait rehab options and cite sources."
```

Flags:
- `--delete-old-agents` snapshot-deletes all existing agents before creating new ones
- `--write-env-updates` writes newly created agent IDs back into `.env` (backs up to `.env.bak`)

Environment toggle (alternative to the flag):

```
WRITE_ENV_ON_CREATE=1
```

## Start script: recreate agents and update `.env`

The provided `start_all.ps1` can optionally delete old agents, recreate the team, and write the new IDs back into your `.env` so the backend reuses them next time.

By default, the script does NOT delete agents. To force recreation and update `.env` (Windows PowerShell):

```powershell
$env:DELETE_OLD_AGENTS="1"
$env:WRITE_ENV_ON_CREATE="1"
pwsh .\start_all.ps1
```

What this does:
- Deletes existing agents, then recreates leader + workers
- Writes the fresh IDs into `.env` (and makes a `.env.bak` backup)
- Starts the FastAPI backend and points you to http://localhost:8000/ui

One‑liner:

```powershell
$env:DELETE_OLD_AGENTS="1"; $env:WRITE_ENV_ON_CREATE="1"; pwsh .\start_all.ps1
```

Notes:
- Ensure `.env` has a valid `PROJECT_ENDPOINT` and, if `librarian_rag` is enabled, `VECTOR_STORE_ID`.
- This will overwrite `LEADER_AGENT_ID` and `AGENT_ID_*` in `.env`; a backup is saved to `.env.bak`.
- If the script cannot find Python/uvicorn on PATH, start the backend directly:

```powershell
C:/Python313/python.exe -m uvicorn scripts.backend:app --reload
```

## Data ingestion (vector store)

Use `scripts/ingest_data.py` (or your own ingestion path) to populate a vector store with PDFs and capture the `VECTOR_STORE_ID`. Ensure `configs/team.yaml` includes a `file_search` tool for `librarian_rag` with your vector store id (or use `${VECTOR_STORE_ID}` substitution and set the env var).

## Telemetry & tracing

- The backend auto-configures Azure Monitor if `APPLICATIONINSIGHTS_CONNECTION_STRING` is set
- If not provided, it attempts to fetch a connection string from the project; if that fails, tracing runs as a no-op
- You can also add OTLP exporters locally (already in requirements)

## Troubleshooting

- Import errors in the editor: run `uv pip install -r requirements.txt` inside your venv
- 401/permission errors: ensure Azure CLI `az login` or VS Code Azure sign-in is active
- Librarian errors: set `VECTOR_STORE_ID` and ensure documents are ingested
- Long runs: backend polls runs up to ~120s; UI shows live events to confirm progress
- Resetting: use the UI “Reset agents” or POST `/reset_agents` with `session_id` and `new_thread=true|false`
 - Live updates: SSE is used by default (EventSource in the browser). If SSE isn’t available, the UI falls back to JSON polling via `/events`.

## Pre-provisioned agent reuse (how it works)

If you already created your agents (leader and/or workers) in Azure AI Projects/Agents, you can tell the backend to reuse them instead of creating new ones on chat init. Set the following environment variables:

- `LEADER_AGENT_ID` — ID of the leader agent
- `AGENT_ID_{WORKER_NAME}` for each worker, where `{WORKER_NAME}` matches the `name` in `configs/team.yaml`, uppercased. For example:
	- `AGENT_ID_LIBRARIAN_RAG`
	- `AGENT_ID_THERAPY_EXPERT`
	- `AGENT_ID_ROBOTICS_EXPERT`

When these are provided:
- The backend uses lightweight handles for the given IDs and skips agent creation.
- The Reset action only deletes agents that were created by this session (tracked internally). Pre-provisioned agents are never deleted.
- If you pre-provision the leader, make sure it already has connected-agent tools wired up to your workers. Otherwise, let the backend create the leader so it can wire the tools for you.

Example `.env` snippet:

```
PROJECT_ENDPOINT=https://<your-project>.<region>.inference.ai.azure.com
VECTOR_STORE_ID=<your-vector-store-id>

# Optional tracing
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=...

# Reuse existing agents
LEADER_AGENT_ID=asst_123456789
AGENT_ID_LIBRARIAN_RAG=asst_lib_abcdef
AGENT_ID_THERAPY_EXPERT=asst_th_abcdef
AGENT_ID_ROBOTICS_EXPERT=asst_ro_abcdef
```

## Directory guide (selected)

- `scripts/backend.py` — FastAPI backend (sessions, runs, streaming, parsing, reset)
- `scripts/run_team.py` — CLI runner for quick checks
	- Supports `--write-env-updates` to persist new agent IDs into `.env`
- `scripts/ingest_data.py` — Example ingestion
- `configs/team.yaml` — Team/agents/tools config (supports `${VECTOR_STORE_ID}` substitution)
- `requirements.txt` — Python deps
- `infra/` — Optional Bicep IaC
- `data/` — Sample materials

## License

See `LICENSE` (if present) for details. Otherwise, content is provided as-is for research and instructional purposes.

## UI semantics (what you’ll see)

- Bubbles labeled as “sender @ receiver” for transparency:
	- Student @ orchestrator (user → leader)
	- orchestrator @ librarian_rag (leader → worker)
	- librarian_rag @ orchestrator (worker → leader)
	- orchestrator @ user (final, adapted)
- Safety and Critic info appended as a system card

## SSE in this app

The UI uses EventSource to subscribe to `/events_sse`. The backend streams lines like `data: {"event_type":"run_status",...}`. It’s a simple, one‑way push from server to browser. If SSE isn’t available, the UI automatically polls `/events`.

## Tests


```powershell
C:/Python313/python.exe -m pip install pytest
C:/Python313/python.exe -m pytest -q tests/test_backend_labeling.py
```

## Start / Stop / Cleanup (recommended)

If you used `start_all.ps1` to start the backend, artifacts may have been written (a `backend.pid` file and `BACKEND_PID` in `.env`). Use the steps below to stop the server and clean up safely.

- Stop the backend (preferred, prompts before force):

```powershell
# prompt for each action; add -Force to skip prompts
pwsh .\stop_all.ps1
```

- Force-stop (non-interactive):

```powershell
# Kills the backend process recorded in .env or backend.pid without prompts
pwsh .\stop_all.ps1 -Force
```

- Manual PID kill (when script doesn't work):

```powershell
# Find candidate processes (look for uvicorn/python running this repo)
Get-Process -Name uvicorn,python | Select-Object Id, ProcessName
# Then kill by PID (replace 12345)
Stop-Process -Id 12345 -Force
```

- Cleanup leftover artifacts (safe to run):

```powershell
# Remove pid file and any BACKEND_PID setting in .env
Remove-Item .\backend.pid -ErrorAction SilentlyContinue
(Get-Content .env) -replace '^[ \t]*BACKEND_PID\s*=.*','' | Set-Content .env
# Remove temporary uploads directory
Remove-Item -Recurse -Force .\tmp_uploads -ErrorAction SilentlyContinue
```

Notes:
- If the backend was started in a terminal (not detached), closing that terminal will also stop the server — but using the script is preferred so the pid file and `.env` are cleaned.
- If you frequently run the backend, consider running it under a process supervisor (e.g., NSSM on Windows or run it in a terminal multiplexer) rather than backgrounding it from a script.

