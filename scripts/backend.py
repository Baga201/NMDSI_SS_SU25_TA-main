"""
backend.py — FastAPI backend for the multi-agent UI.

Key behaviors
- Keeps one Azure Agents thread per session_id and caches created agents.
- Loads team (leader + workers) from YAML; substitutes ${VECTOR_STORE_ID}.
- Gives librarian_rag a FileSearchTool with definitions + resources.
- Posts the user message, runs the leader (create_and_process if available).
- Streams run events correctly (AgentRunStream.iter_events()) for the UI panel.
- Builds a WhatsApp-style chat transcript with @mentions:
    user -> orchestrator -> @worker responses -> orchestrator final
- Parses worker JSON; only 'result' is required, warns if thought/action absent.
- Surfaces connected-agent tool outputs as separate chat bubbles.
- Optional OpenTelemetry tracing (safe no-ops if not installed).
"""

import os
import re
import json
import time
import threading
from typing import Dict, Any, List, Optional
from types import SimpleNamespace


import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Form, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import (
    MessageRole,
    ListSortOrder,
    FileSearchTool,
    ConnectedAgentTool,
)

# Optional image/content block classes (SDK-version tolerant)
try:
    from azure.ai.agents.models import (
        MessageInputTextBlock,
        MessageInputImageFileBlock,
        MessageImageFileParam,
        FilePurpose,
    )
    IMAGE_API_AVAILABLE = True
except Exception:
    IMAGE_API_AVAILABLE = False

# -------------------- Optional: Telemetry (graceful fallback) --------------------
tracer = None
try:
    from opentelemetry import trace
    tracer = trace.get_tracer(__name__)
except Exception:
    tracer = None
# -----------------------------------------------------------------------------

load_dotenv(override=True)

CONFIG_PATH = os.getenv("TEAM_CONFIG_PATH", "configs/team.yaml")
ENDPOINT = os.getenv("PROJECT_ENDPOINT")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")
credential = DefaultAzureCredential()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# In-memory store for demo/dev: { session_id: {thread, agents, leader, run_events, ...} }
sessions: Dict[str, Dict[str, Any]] = {}
SESS_TTL_SECS = int(os.getenv("SESSION_TTL_SECS", "1800"))  # default 30 minutes

# ----------------------------- Utilities ---------------------------------------

def load_team_config(path: str) -> Dict[str, Any]:
    """Load YAML and substitute ${VECTOR_STORE_ID} if present."""
    # Re-read .env on each call so updates to IDs are reflected without restarting the server
    try:
        load_dotenv(override=True)
    except Exception:
        pass
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    team_cfg = cfg.get("team", {})
    if "workers" not in team_cfg and "workers" in cfg:
        team_cfg["workers"] = cfg["workers"]

    vsid = os.getenv("VECTOR_STORE_ID")
    for w in team_cfg.get("workers", []):
        for tool in w.get("tools", []):
            if isinstance(tool, dict) and "vector_store_ids" in tool and vsid:
                tool["vector_store_ids"] = [
                    (vsid if str(v).strip() == "${VECTOR_STORE_ID}" else v)
                    for v in tool["vector_store_ids"]
                ]
    # Allow overriding with env-provided agent ids
    # LEADER_AGENT_ID for leader, and AGENT_ID_<WORKER_NAME> per worker
    leader_id = os.getenv("LEADER_AGENT_ID")
    if leader_id:
        team_cfg["leader_id"] = leader_id

    for w in team_cfg.get("workers", []):
        env_key = f"AGENT_ID_{str(w.get('name','')).upper()}"
        wid = os.getenv(env_key)
        if wid:
            w["id"] = wid
    return team_cfg


# ----------------------------- Evaluator & Safety -----------------------------
def simple_safety_check(text: str) -> Dict[str, Any]:
    """Very lightweight safety check. Replace with Azure Content Safety if desired.
    Returns {status: 'ok'|'flagged', reasons: [..]}.
    """
    reasons = []
    lowered = (text or "").lower()
    risky = ["self-harm", "suicide", "kill myself", "harm others", "violence"]
    for term in risky:
        if term in lowered:
            reasons.append(f"contains: {term}")
    status = "ok" if not reasons else "flagged"
    return {"status": status, "reasons": reasons}


def critic_evaluate(result_text: str, used_workers: List[str], had_citations: bool) -> Dict[str, Any]:
    """Heuristic evaluator to reduce cost and avoid extra LLM calls.
    Returns a dict with hallucination_risk, missing_citations, confidence, notes.
    """
    txt = (result_text or "").strip()
    length = len(txt)
    # naive signals
    missing_citations = ("librarian_rag" in used_workers) and (not had_citations)
    hallucination_risk = missing_citations or (length > 2000)
    base_conf = 0.7
    if missing_citations:
        base_conf -= 0.2
    if length < 200:
        base_conf += 0.05
    elif length > 1200:
        base_conf -= 0.05
    base_conf = max(0.0, min(1.0, base_conf))
    notes = []
    if missing_citations:
        notes.append("Worker `librarian_rag` used but no citations detected.")
    return {
        "hallucination_risk": bool(hallucination_risk),
        "missing_citations": bool(missing_citations),
        "confidence": base_conf,
        "notes": notes,
    }


# ----------------------------- Memory / Summarization -------------------------
def update_session_summary(sess: Dict[str, Any], chat_history: List[Dict[str, Any]], max_len: int = 800) -> None:
    """Maintain a compact rolling summary. This is a cheap heuristic fallback.
    It trims and stores a brief summary under sess['summary'].
    """
    if not chat_history:
        return
    # simple: take last few messages and truncate
    snippets = []
    for e in chat_history[-6:]:
        who = e.get("agent") or e.get("role")
        text = (e.get("text") or "").strip().replace("\n", " ")
        if text:
            snippets.append(f"{who}: {text}")
    joined = " | ".join(snippets)
    if len(joined) > max_len:
        joined = joined[: max_len - 1] + "…"
    sess["summary"] = joined


def adapt_for_role(text: str, role: Optional[str]) -> Dict[str, Any]:
    """
    Lightweight role adapter that rewrites the final leader->user message for the target audience.
    Returns a dict with keys: { adapted_text, role, strategy }.
    """
    role = (role or "").strip() or "Student"
    base = (text or "").strip()
    if not base:
        return {"adapted_text": base, "role": role, "strategy": "none"}

    # naive sentence split
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", base) if p.strip()]
    # helpers
    def bullets(ps: List[str], limit: int = 6) -> str:
        ps2 = ps[:limit]
        return "\n".join([f"• {p}" for p in ps2])

    if role.lower() in {"executive", "exec", "leader"}:
        adapted = (
            "Executive summary:\n"
            + bullets(parts, 5)
            + "\n\nKey risks & mitigations:\n• Risk: Gaps or uncertainty remain — Mitigation: follow-up with librarian for citations, validate assumptions.\n"
            + "\nNext steps:\n• Decide scope\n• Assign owner\n• Set timeline"
        )
        return {"adapted_text": adapted, "role": role, "strategy": "bulleted_summary"}

    if role.lower() in {"engineer", "developer", "technical"}:
        adapted = (
            "Technical brief:\n"
            + bullets(parts, 8)
            + "\n\nImplementation notes:\n• Inputs/outputs clarified\n• Constraints and edge cases noted\n• Add tests for parsing and delegation mapping"
        )
        return {"adapted_text": adapted, "role": role, "strategy": "technical_brief"}

    if role.lower() in {"therapist", "therapy", "clinician"}:
        adapted = (
            "Supportive guidance (not medical advice):\n"
            + bullets(parts, 8)
            + "\n\nIf this concerns wellbeing or safety, consider contacting a trusted professional or local support resources."
        )
        return {"adapted_text": adapted, "role": role, "strategy": "supportive_guidance"}

    if role.lower() in {"robotics specialist", "robotics", "roboticist"}:
        adapted = (
            "Robotics-focused summary:\n"
            + bullets(parts, 8)
            + "\n\nConsider control, sensing, and integration aspects; verify with simulation/bench tests."
        )
        return {"adapted_text": adapted, "role": role, "strategy": "domain_summary"}

    # Default: Student/learner
    adapted = (
        "Explainer:\n"
        + bullets(parts, 8)
        + "\n\nThink of it like this: we're breaking the problem into clear steps and checking our sources."
    )
    return {"adapted_text": adapted, "role": role, "strategy": "explainer"}


def ensure_session(session_id: Optional[str] = None) -> str:
    """Create a new thread-backed session if none or unknown session_id."""
    with AIProjectClient(endpoint=ENDPOINT, credential=credential) as pc:
        agents = pc.agents
        if not session_id or session_id not in sessions:
            thread = agents.threads.create()
            sid = thread.id
            sessions[sid] = {"thread": thread, "agents": {}, "leader": None, "last_seen": time.time()}
            return sid
        sessions[session_id]["last_seen"] = time.time()
        return session_id


def build_agents_for_session(session_id: str, team_cfg: Dict[str, Any]) -> None:
    """Create workers and leader for a session once; cache them."""
    sess = sessions[session_id]
    # If agents are already cached, verify they match current team_cfg IDs from .env; if mismatched, rebuild
    if sess.get("leader") and sess.get("agents"):
        try:
            cached_leader_id = getattr(sess.get("leader"), "id", None)
            desired_leader_id = team_cfg.get("leader_id")
            mismatch = False
            if desired_leader_id and cached_leader_id and desired_leader_id != cached_leader_id:
                mismatch = True
            # Check workers
            if not mismatch:
                for w in team_cfg.get("workers", []):
                    wname = w.get("name")
                    desired_wid = w.get("id")
                    if not desired_wid:
                        continue
                    cached = (sess.get("agents") or {}).get(wname)
                    cached_wid = getattr(cached, "id", None) if cached else None
                    if cached_wid and desired_wid and cached_wid != desired_wid:
                        mismatch = True
                        break
            if not mismatch:
                return  # already built and matches
            # Else: reset cache to rebuild with new IDs
            sess["agents"] = {}
            sess["leader"] = None
        except Exception:
            # On any error, fall through to rebuild
            sess["agents"] = {}
            sess["leader"] = None

    managed_ids: set = set()
    with AIProjectClient(endpoint=ENDPOINT, credential=credential) as pc:
        client = pc.agents

        # 1) Workers: reuse if id provided, else create and mark managed
        workers = {}
        for w in team_cfg.get("workers", []):
            name = w["name"]
            existing_id = w.get("id")
            if existing_id:
                # create lightweight handle; no network dependency
                agent = SimpleNamespace(id=existing_id, name=name)
            else:
                if name == "librarian_rag":
                    vsids = None
                    for tool in w.get("tools", []):
                        if isinstance(tool, dict) and tool.get("type") == "file_search":
                            vsids = tool.get("vector_store_ids") or ([VECTOR_STORE_ID] if VECTOR_STORE_ID else None)
                            break
                    if not vsids:
                        raise ValueError("VECTOR_STORE_ID missing for librarian_rag; provide env or config.")

                    fs = FileSearchTool(vector_store_ids=vsids)
                    agent = client.create_agent(
                        model=w["model"],
                        name=name,
                        instructions=w["instructions"],
                        tools=fs.definitions,
                        tool_resources=fs.resources,
                    )
                else:
                    agent = client.create_agent(
                        model=w["model"],
                        name=name,
                        instructions=w["instructions"],
                    )
                managed_ids.add(getattr(agent, "id", None))
            workers[name] = agent

        # 2) Leader with connected agents: reuse if id provided
        leader_id = team_cfg.get("leader_id")
        if leader_id:
            leader = SimpleNamespace(id=leader_id, name=team_cfg.get("leader_name", "leader"))
        else:
            leader_tools = []
            for n, a in workers.items():
                cat = ConnectedAgentTool(id=a.id, name=n, description=f"Delegated agent: {n}")
                leader_tools.append(cat.definitions[0])

            leader = client.create_agent(
                model=team_cfg["leader_model"],
                name=team_cfg["leader_name"],
                instructions=team_cfg["leader_instructions"],
                tools=leader_tools,
            )
            managed_ids.add(getattr(leader, "id", None))

    sess["agents"] = workers
    sess["leader"] = leader
    sess["leader_name"] = team_cfg.get("leader_name", "orchestrator")
    sess["managed_agent_ids"] = managed_ids
    sess["last_seen"] = time.time()


def parse_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from text (tolerant to fences and extra words)."""
    # Direct JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fenced code
    s = text.strip()
    if s.startswith("```"):
        s2 = s.strip("`")
        s2 = s2.split("\n", 1)[1] if "\n" in s2 else s2
        try:
            obj = json.loads(s2)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # First {...}
    m = re.search(r"\{[\s\S]*?\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def validate_structured_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flexible validation:
    - required: result
    - recommended: thought, action
    """
    required = {"result"}
    recommended = {"thought", "action"}
    out = {
        "structured_valid": all(k in payload for k in required),
        "structured_missing": [k for k in required if k not in payload],
    }
    missing_recommended = [k for k in recommended if k not in payload]
    if missing_recommended:
        out["structured_warnings"] = {"missing_recommended": missing_recommended}
    return out


def _stream_run_events(thread_id: str, run_id: str, session_id: str) -> None:
    """
    Deterministic polling of run status for live UI; avoids SDK streaming API differences.
    Appends concise status changes and exits on terminal state.
    """
    try:
        with AIProjectClient(endpoint=ENDPOINT, credential=credential) as pc:
            client = pc.agents
            last_status = None
            # Track seen message ids to emit new agent message events
            seen_ids = sessions.setdefault(session_id, {}).setdefault("_seen_msg_ids", set())
            start_ts = time.time()
            seen_steps = sessions.setdefault(session_id, {}).setdefault("_seen_step_ids", set())
            timeout_s = 180
            # Build reverse map agent_id -> worker name for readability
            try:
                sess = sessions.get(session_id, {})
                id_to_name = {getattr(a, 'id', None): n for n, a in (sess.get('agents') or {}).items()}
            except Exception:
                id_to_name = {}
            while True:
                run = client.runs.get(thread_id=thread_id, run_id=run_id)
                status = getattr(run, "status", None)
                if status != last_status:
                    record = {"event_type": "run_status", "event_data": str(status)}
                    sessions.setdefault(session_id, {}).setdefault("run_events", []).append(record)
                    last_status = status
                # Poll messages for near-real-time agent communications
                try:
                    msgs = client.messages.list(thread_id=thread_id, order=ListSortOrder.ASCENDING)
                    for m in msgs:
                        mid = getattr(m, "id", None)
                        if not mid or mid in seen_ids:
                            continue
                        seen_ids.add(mid)
                        who = getattr(m, "agent_name", None) or getattr(m, "sender", None) or m.role
                        if who and str(m.role).lower() == "agent":
                            sessions.setdefault(session_id, {}).setdefault("run_events", []).append({
                                "event_type": "agent_message",
                                "event_data": {"agent": who, "id": mid},
                            })
                except Exception:
                    pass
                # Emit delegation/reply based on steps
                try:
                  steps = getattr(run, "steps", None) or []
                  for st in steps:
                      sid = getattr(st, "id", None) or getattr(st, "step_id", None) or str(st)
                      if sid in seen_steps:
                          continue
                      seen_steps.add(sid)
                      tool_name = None
                      try:
                          tc = getattr(st, "tool_call", None) or getattr(st, "tool_invocation", None)
                          if tc is not None:
                              tool_name = getattr(tc, "tool_name", None) or getattr(tc, "name", None)
                      except Exception:
                          pass
                      if tool_name:
                          # Emit a short typing indicator for the target worker, then emit the delegation
                          sessions.setdefault(session_id, {}).setdefault("run_events", []).append({
                              "event_type": "typing",
                              "event_data": {"agent": tool_name, "typing": True, "duration": 800}
                          })
                          # small pause so UI shows typing briefly
                          try:
                              time.sleep(0.6)
                          except Exception:
                              pass
                          sessions.setdefault(session_id, {}).setdefault("run_events", []).append({
                              "event_type": "delegation",
                              "event_data": {"from": sessions.get(session_id, {}).get("leader_name", "leader"), "to": tool_name},
                          })
                      elif getattr(st, "text", None):
                          # Treat as a worker reply text — emit typing then reply to give UX sense
                          who = id_to_name.get(getattr(st, "agent_id", None), getattr(st, "agent_id", None))
                          sessions.setdefault(session_id, {}).setdefault("run_events", []).append({
                              "event_type": "typing",
                              "event_data": {"agent": who, "typing": True, "duration": 900}
                          })
                          try:
                              time.sleep(0.6)
                          except Exception:
                              pass
                          sessions.setdefault(session_id, {}).setdefault("run_events", []).append({
                              "event_type": "reply",
                              "event_data": {"from": who, "snippet": str(getattr(st, "text", ""))[:120]},
                          })
                except Exception:
                  pass
                if status in ["completed", "failed", "cancelled"]:
                    sessions.setdefault(session_id, {})["run_complete"] = True
                    break
                if time.time() - start_ts > timeout_s:
                    sessions.setdefault(session_id, {})["run_events"].append({"event_type": "timeout", "event_data": "run polling timeout"})
                    sessions.setdefault(session_id, {})["run_complete"] = True
                    break
                time.sleep(0.5)
    except Exception as e:
        sessions.setdefault(session_id, {}).setdefault("run_events", []).append({"event_type": "poll_exception", "event_data": str(e)})
        sessions.setdefault(session_id, {})["run_complete"] = True

# ----------------------------- Endpoints ---------------------------------------

@app.get("/")
async def root():
    return {"status": "ok", "endpoints": ["/session", "/chat", "/events", "/reset_agents"]}


@app.post("/session")
async def create_session():
    sid = ensure_session(None)
    return {"session_id": sid}


@app.post("/chat")
async def chat(
    session_id: str = Form(None),
    message: str = Form(...),
    role: str = Form(None),
    image: UploadFile = File(None),
):
    # 1) Ensure session & team
    sid = ensure_session(session_id)
    team_cfg = load_team_config(CONFIG_PATH)
    build_agents_for_session(sid, team_cfg)

    sess = sessions[sid]
    # Persist role in session for consistent adaptation across turns
    if role:
        sess["role"] = role
    leader = sess["leader"]
    thread = sess["thread"]

    # 2) Post user message (optionally include uploaded image via content blocks)
    with AIProjectClient(endpoint=ENDPOINT, credential=credential) as pc:
        client = pc.agents
        # If we have a rolling summary, prepend it as context hint
        summary_hint = sessions[sid].get("summary")
        # include UI role as a prefix so agents can use it (metadata support varies by SDK)
        base_text = f"[role: {role}]\n{message}" if role else message
        user_text = (f"[summary]\n{summary_hint}\n\n" + base_text) if summary_hint else base_text
        posted = False
        # If image uploaded and SDK supports content blocks, upload and include image in message
        try:
            if IMAGE_API_AVAILABLE and image is not None and getattr(image, "filename", ""):
                # persist to temp file (SDK upload uses file_path)
                tmp_dir = os.path.join(os.getcwd(), "tmp_uploads")
                os.makedirs(tmp_dir, exist_ok=True)
                tmp_path = os.path.join(
                    tmp_dir, f"{int(time.time()*1000)}_" + os.path.basename(image.filename)
                )
                # UploadFile.read() is awaitable in async endpoint
                raw = await image.read()
                with open(tmp_path, "wb") as fh:
                    fh.write(raw)

                try:
                    up = client.files.upload_and_poll(file_path=tmp_path, purpose=FilePurpose.AGENTS)
                    file_param = MessageImageFileParam(file_id=up.id, detail="high")
                    content_blocks = [
                        MessageInputTextBlock(text=user_text),
                        MessageInputImageFileBlock(image_file=file_param),
                    ]
                    client.messages.create(
                        thread_id=thread.id, role=MessageRole.USER, content=content_blocks
                    )
                    posted = True
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
        except Exception:
            # fall back to text-only if anything fails
            posted = False

        if not posted:
            client.messages.create(thread_id=thread.id, role=MessageRole.USER, content=user_text)

        # 3) Run leader deterministically (create, then poll status)
        if tracer:
            span_ctx = tracer.start_as_current_span("leader.run")
            span_ctx.__enter__()
        try:
            run = client.runs.create(thread_id=thread.id, agent_id=leader.id)
        finally:
            if tracer:
                try:
                    span_ctx.__exit__(None, None, None)
                except Exception:
                    pass

        # 4) Start background event stream (fresh buffer)
        sess["run_events"] = []
        sess["run_complete"] = False
        sess["run_id"] = getattr(run, "id", None)
        t = threading.Thread(target=_stream_run_events, args=(thread.id, getattr(run, "id", None), sid), daemon=True)
        t.start()

    # 5) Poll run until terminal status (keeps API simple for UI)
        poll = run
        start_ts = time.time()
        timeout_s = 180
        while getattr(poll, "status", "") not in ["completed", "failed", "cancelled"]:
            if time.time() - start_ts > timeout_s:
                break
            time.sleep(0.5)
            poll = client.runs.get(thread_id=thread.id, run_id=run.id)
        full_run = poll

    sess["last_seen"] = time.time()

    # 6) Build chat history with provenance
    with AIProjectClient(endpoint=ENDPOINT, credential=credential) as pc:
        client = pc.agents
        msgs = client.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)

        chat_history: List[Dict[str, Any]] = []
        delegation_chain: List[str] = []
        citations: List[str] = []

        # Friendly name mapping for agents
        agent_id_to_name = {getattr(a, "id", None): n for n, a in sess["agents"].items()}
        leader_name = team_cfg.get("leader_name", "orchestrator")
        try:
            lid = getattr(sess.get("leader"), "id", None)
            if lid:
                agent_id_to_name[lid] = leader_name
        except Exception:
            pass

        def _norm_role(val: Any) -> str:
            s = str(val).lower()
            if "user" in s:
                return "user"
            if "system" in s:
                return "system"
            if "agent" in s:
                return "agent"
            return s or "agent"

        for m in msgs:
            if not m.text_messages:
                continue

            last_text = m.text_messages[-1]
            text = last_text.text.value
            # Normalize sender label
            role_norm = _norm_role(getattr(m, "role", None))
            mid = getattr(m, "agent_id", None)
            if role_norm == "user":
                agent_label = sess.get("role") or "user"
            elif role_norm == "system":
                agent_label = "system"
            else:
                agent_label = agent_id_to_name.get(mid) or leader_name
            prov = {
                "role": role_norm,
                "agent": agent_label,
                "text": text,
                "is_final": False,
                "ts": getattr(m, "created_at", None),
            }
            # user @ leader mention to reflect direction
            if role_norm == "user":
                prov["delegated_to"] = leader_name

            # Parse worker JSON (tolerant)
            parsed = parse_first_json_object(text)
            if isinstance(parsed, dict):
                prov["structured"] = True
                prov["structured_json"] = parsed
                prov.update(validate_structured_payload(parsed))
            else:
                prov["structured"] = False

            # Inline citations if SDK annotations exist (file + URL)
            anns = getattr(last_text, "annotations", None)
            if anns:
                for a in anns:
                    if hasattr(a, "file_citation"):
                        fid = a.file_citation.file_id
                        try:
                            text = text.replace(a.text, f" [{fid}]")
                            prov["text"] = text
                        except Exception:
                            pass
                        citations.append(fid)
            # URL citation annotations may live on the message itself
            url_anns = getattr(m, "url_citation_annotations", None)
            if url_anns:
                try:
                    placeholders = {
                        ua.text: f" [see {ua.url_citation.title}]({ua.url_citation.url})"
                        for ua in url_anns
                        if getattr(ua, "url_citation", None)
                    }
                    if placeholders:
                        for k, v in placeholders.items():
                            if k and k in text:
                                text = text.replace(k, v)
                        prov["text"] = text
                except Exception:
                    pass

            chat_history.append(prov)

            # Track a simple chain: every agent message appends its author
            if role_norm == "agent":
                delegation_chain.append(agent_label)

    # Mark final agent message (pick the last agent entry in chat_history)
    final_agent = None
    if delegation_chain:
        final_agent = delegation_chain[-1]
    else:
        # Fallback: identify any last agent entry as final
        for ent in reversed(chat_history):
            if ent.get("role", "").lower() == "agent":
                final_agent = ent.get("agent")
                ent["is_final"] = True
                if ent.get("agent") == team_cfg.get("leader_name", "orchestrator"):
                    ent["delegated_to"] = "user"
                break
    if final_agent:
        for ent in reversed(chat_history):
            if ent.get("agent") == final_agent and ent.get("role", "").lower() == "agent":
                ent["is_final"] = True
                if ent.get("agent") == team_cfg.get("leader_name", "orchestrator"):
                    ent["delegated_to"] = "user"
                break

    # 7) Serialize run steps/activities for a readable trace (always compute)
    def serial_step(step: Any) -> Dict[str, Any]:
        s: Dict[str, Any] = {}
        try:
            s["step_type"] = getattr(step, "type", getattr(step, "step_type", None))
        except Exception:
            s["step_type"] = None
        s["agent_id"] = getattr(step, "agent_id", None) or getattr(step, "actor", None)
        tool_name = None
        try:
            tc = getattr(step, "tool_call", None) or getattr(step, "tool_invocation", None)
            if tc is not None:
                tool_name = getattr(tc, "tool_name", None) or getattr(tc, "name", None)
                s["tool_input"] = getattr(tc, "input", getattr(tc, "inputs", None))
                s["tool_output"] = getattr(tc, "output", getattr(tc, "outputs", None))
        except Exception:
            pass
        s["tool_name"] = tool_name
        # text-ish
        try:
            tv = getattr(step, "text", None) or getattr(step, "content", None) or getattr(step, "value", None)
            if hasattr(tv, "value"):
                tv = tv.value
            s["text"] = tv
        except Exception:
            s["text"] = None
        s["status"] = getattr(step, "status", None)
        s["timestamp"] = getattr(step, "timestamp", None)
        try:
            s["repr"] = str(step)
        except Exception:
            s["repr"] = None
        # tracing
        if tracer:
            try:
                with tracer.start_as_current_span("agent.step") as sp:
                    if s.get("step_type"):
                        sp.set_attribute("step.type", str(s["step_type"]))
                    if s.get("agent_id"):
                        sp.set_attribute("agent.id", str(s["agent_id"]))
                    if s.get("tool_name"):
                        sp.set_attribute("tool.name", str(s["tool_name"]))
                        sp.set_attribute("gen_ai.tool.name", str(s["tool_name"]))
                    if s.get("status"):
                        sp.set_attribute("step.status", str(s["status"]))
                    sp.set_attribute("gen_ai.system", "azure.ai.agents")
                    if getattr(full_run, "id", None):
                        sp.set_attribute("run.id", str(getattr(full_run, "id", "")))
                    sp.set_attribute("session.id", sid)
            except Exception:
                pass
        return s

    run_trace: List[Dict[str, Any]] = []
    if getattr(full_run, "steps", None):
        for st in full_run.steps:
            run_trace.append(serial_step(st))
    elif getattr(full_run, "activities", None):
        for ac in full_run.activities:
            run_trace.append(serial_step(ac))
    else:
        try:
            run_trace.append({"repr": str(full_run)})
        except Exception:
            pass

    # 8) Surface explicit leader→worker and worker→leader messages based on run_trace
    agent_id_to_name = {getattr(a, "id", None): n for n, a in sess["agents"].items()}
    name_lookup = {str(n).lower(): n for n in sess["agents"].keys()}
    leader_name = team_cfg.get("leader_name", "orchestrator")
    dialogue_entries: List[Dict[str, Any]] = []

    used_workers: List[str] = []
    for st in run_trace:
        try:
            tname = (st.get("tool_name") or "").lower()
            repr_txt = (st.get("repr") or "")
            st_agent_id = st.get("agent_id")

            # Identify worker for this step
            worker_name = None
            if st_agent_id and st_agent_id in agent_id_to_name:
                worker_name = agent_id_to_name[st_agent_id]
            elif tname and tname in name_lookup:
                worker_name = name_lookup[tname]
            elif "connected_agent" in repr_txt.lower():
                m = re.search(r"(asst_[A-Za-z0-9_\-]+)", repr_txt)
                if m:
                    worker_name = agent_id_to_name.get(m.group(1))

            if not worker_name:
                continue

            # Leader -> worker (delegation prompt)
            tin = st.get("tool_input")
            if tin:
                if isinstance(tin, (dict, list)):
                    try:
                        tin_text = json.dumps(tin)
                    except Exception:
                        tin_text = str(tin)
                else:
                    tin_text = str(tin)
                if worker_name not in used_workers:
                    used_workers.append(worker_name)
                dialogue_entries.append({
                    "role": "agent",
                    "agent": leader_name,
                    "delegated_to": worker_name,
                    "tool": tname or None,
                    "text": tin_text,
                    "is_final": False,
                    "structured": False,
                    "ts": st.get("timestamp"),
                })

            # Worker -> leader (reply)
            tout = st.get("tool_output") or st.get("text")
            if tout:
                if isinstance(tout, (dict, list)):
                    try:
                        tout_text = json.dumps(tout)
                    except Exception:
                        tout_text = str(tout)
                else:
                    tout_text = str(tout)
                parsed = parse_first_json_object(tout_text)
                ent: Dict[str, Any] = {
                    "role": "agent",
                    "agent": worker_name,
                    "delegated_from": leader_name,
                    "tool": tname or None,
                    "text": tout_text,
                    "is_final": False,
                    "ts": st.get("timestamp"),
                }
                if isinstance(parsed, dict):
                    ent["structured"] = True
                    ent["structured_json"] = parsed
                    ent.update(validate_structured_payload(parsed))
                else:
                    ent["structured"] = False
                dialogue_entries.append(ent)
        except Exception:
            continue

    if dialogue_entries:
        # Insert dialogue entries right after the first leader message
        insert_at = len(chat_history)
        for i, e in enumerate(chat_history):
            if e.get("agent") == leader_name:
                insert_at = i + 1
                break
        chat_history[insert_at:insert_at] = dialogue_entries

    # 8.1) Safety + evaluator (heuristic, local)
    final_text = None
    for ent in reversed(chat_history):
        if ent.get("is_final"):
            final_text = ent.get("text")
            break
    # had citations?
    had_citations = any("[see " in (final_text or "") or "[" in (final_text or "") for _ in [0])
    safety = simple_safety_check(final_text or "")
    critic = critic_evaluate(final_text or "", used_workers, had_citations)

    # 9) Basic run info (SDK-version tolerant)
    run_info: Dict[str, Any] = {
        "status": getattr(full_run, "status", None),
        "id": getattr(full_run, "id", None),
    }
    for attr in ["model", "last_error", "created_at", "completed_at", "usage", "metrics", "metadata"]:
        try:
            val = getattr(full_run, attr, None)
            if val is not None:
                run_info[attr] = val
        except Exception:
            pass
    run_info["safety"] = safety
    run_info["critic"] = critic

    # 10) Role adaptation: rewrite final leader->user message for target role
    target_role = sess.get("role") or role
    adapted = adapt_for_role(final_text or "", target_role)
    run_info["role_adapter"] = adapted
    # Inject adapted bubble replacing final leader->user text, but keep original as details
    if final_text:
        for ent in reversed(chat_history):
            if ent.get("is_final"):
                ent["original_text"] = ent.get("text")
                ent["text"] = adapted.get("adapted_text", ent["original_text"])
                ent["adapted_for"] = adapted.get("role")
                ent["adapt_strategy"] = adapted.get("strategy")
                break

    # 10.1) If the final leader message contains a structured payload describing
    # delegations and worker_responses, expand it into a sequence of chat bubbles
    # so the UI renders an orchestrator->worker conversation (messenger style).
    try:
        # Find the final leader entry (after adaptation replacement above)
        final_entry = None
        for ent in reversed(chat_history):
            if ent.get("is_final") and ent.get("agent") == leader_name:
                final_entry = ent
                break
        if final_entry:
            # Try structured_json on the final entry; fallback to parsing original text
            structured = final_entry.get("structured_json")
            if not structured:
                structured = parse_first_json_object(final_entry.get("original_text") or final_entry.get("text") or "")

            if isinstance(structured, dict) and (structured.get("delegations") or structured.get("worker_responses") or structured.get("final_recommendation")):
                # Build a sequence of synthetic dialogue entries
                seq = []
                # Orchestrator explainer / thought
                leader_thought = structured.get("leader_thought") or structured.get("explainer") or None
                if leader_thought:
                    seq.append({
                        "role": "agent",
                        "agent": leader_name,
                        "text": leader_thought,
                        "is_final": False,
                        "ts": final_entry.get("ts"),
                    })

                # Delegations: present each delegation as leader -> worker bubble
                for d in structured.get("delegations", []) or []:
                    try:
                        to = d.get("to") or d.get("worker") or d.get("agent") or ""
                        prompt_text = d.get("prompt") or d.get("subtask") or (d.get("task") if isinstance(d.get("task"), str) else None) or str(d)
                    except Exception:
                        to = ""
                        prompt_text = str(d)
                    seq.append({
                        "role": "agent",
                        "agent": leader_name,
                        "delegated_to": to,
                        "text": prompt_text,
                        "is_final": False,
                        "ts": final_entry.get("ts"),
                    })

                # Worker responses: present each worker reply as a bubble from that worker
                for w in structured.get("worker_responses", []) or []:
                    try:
                        w_from = w.get("from") or w.get("agent") or w.get("name") or None
                        # Build readable text from structured response dict
                        if isinstance(w, dict):
                            # remove 'from' key when serializing
                            serial_items = []
                            for kk, vv in w.items():
                                if kk == "from":
                                    continue
                                if isinstance(vv, (dict, list)):
                                    serial_items.append(f"{kk}: {json.dumps(vv)}")
                                else:
                                    serial_items.append(f"{kk}: {vv}")
                            reply_text = "\n".join(serial_items) if serial_items else str(w)
                        else:
                            reply_text = str(w)
                    except Exception:
                        w_from = None
                        reply_text = str(w)
                    seq.append({
                        "role": "agent",
                        "agent": w_from or "worker",
                        "delegated_from": leader_name,
                        "text": reply_text,
                        "structured": isinstance(w, dict),
                        "structured_json": (w if isinstance(w, dict) else None),
                        "is_final": False,
                        "ts": final_entry.get("ts"),
                    })

                # Final recommendation bubble (leader summarises)
                final_rec = structured.get("final_recommendation") or structured.get("final") or None
                if final_rec:
                    seq.append({
                        "role": "agent",
                        "agent": leader_name,
                        "text": final_rec,
                        "is_final": True,
                        "ts": final_entry.get("ts"),
                    })

                # Replace the original final_entry in chat_history with the sequence
                # Find its index and splice
                idx = None
                for i, c in enumerate(chat_history):
                    if c is final_entry:
                        idx = i
                        break
                if idx is not None:
                    # remove the original final entry and insert the seq
                    chat_history[idx:idx+1] = seq
    except Exception:
        # If anything fails, leave chat_history unchanged
        pass

    # 11) Add a compact system card to explain backend signals
    try:
        sys_text = (
            f"Workers used: {', '.join(used_workers) or '—'}\n"
            f"Safety: {safety.get('status')}"
            + (f" — {', '.join(safety.get('reasons', []))}" if safety.get('reasons') else "")
            + f"\nCritic: confidence={critic.get('confidence'):.2f}, hallucination_risk={critic.get('hallucination_risk')}"
        )
        chat_history.append({
            "role": "system",
            "agent": "system",
            "text": sys_text,
            "is_final": False,
            "ts": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        })
    except Exception:
        pass

    # 12) Update rolling summary for memory
    try:
        update_session_summary(sess, chat_history)
    except Exception:
        pass

    return {
        "session_id": sid,
        "run_id": getattr(run, "id", None),
        "final_agent": final_agent,
        "delegation_chain": delegation_chain,
        "citations": citations,
        "chat_history": chat_history,
        "run_trace": run_trace,
        "run_info": run_info,
        "agent_map": agent_id_to_name,
    }


@app.get("/events")
async def get_events(session_id: str):
    sess = sessions.get(session_id, {})
    if sess:
        sess["last_seen"] = time.time()
    return {
        "run_events": sess.get("run_events", []),
        "run_complete": sess.get("run_complete", False),
        "run_id": sess.get("run_id"),
    }


@app.post("/reset_agents")
async def reset_agents(
    session_id: Optional[str] = Form(None),
    new_thread: bool = Form(False),
):
    """
    Delete the leader and worker agents created for this session and reset the session state.
    Optionally create a fresh thread for this session while keeping the same session_id key.
    """
    sid = ensure_session(session_id)
    sess = sessions.get(sid)
    if not sess:
        return {"error": "session not found", "session_id": sid}

    deleted = {"leader": None, "workers": []}
    with AIProjectClient(endpoint=ENDPOINT, credential=credential) as pc:
        client = pc.agents

        # Delete leader first (if exists)
        leader = sess.get("leader")
        managed_ids = sess.get("managed_agent_ids", set())
        if leader is not None and getattr(leader, "id", None) in managed_ids:
            try:
                client.delete_agent(getattr(leader, "id", leader))
                deleted["leader"] = getattr(leader, "id", leader)
            except Exception:
                pass

        # Delete workers
        workers = sess.get("agents", {}) or {}
        # snapshot ids
        worker_ids = []
        try:
            for a in workers.values():
                worker_ids.append(getattr(a, "id", a))
        except Exception:
            pass
        for aid in reversed(worker_ids):
            if aid in managed_ids:
                try:
                    client.delete_agent(aid)
                    deleted["workers"].append(aid)
                except Exception:
                    pass

        # Optionally create a new thread; keep session_id the same
        new_thread_obj = None
        if new_thread:
            try:
                new_thread_obj = client.threads.create()
                sess["thread"] = new_thread_obj
            except Exception:
                new_thread_obj = None

    # Clear cached agents and run state
    sess["agents"] = {}
    sess["leader"] = None
    sess["run_events"] = []
    sess["run_complete"] = False
    sess["run_id"] = None

    return {
        "session_id": sid,
        "reset": "ok",
        "deleted": deleted,
        "thread_id": getattr(sess.get("thread"), "id", None),
        "new_thread": bool(new_thread),
    }


# ----------------------------- Simple HTML UI (Jinja2 + HTMX) -----------------

@app.get("/health")
async def health():
    return {"status": "ok", "session_count": len(sessions)}

@app.get("/ui", response_class=HTMLResponse)
async def ui_page(request: Request):
    sid = ensure_session(None)
    return templates.TemplateResponse(
        "ui.html",
        {
            "request": request,
            "session_id": sid,
            "chat_history": [],
        },
    )


@app.post("/ui/chat", response_class=HTMLResponse)
async def ui_chat(
    request: Request,
    session_id: str = Form(None),
    message: str = Form(""),
    role: str = Form(None),
    image: UploadFile = File(None),
):
    # Delegate to JSON chat handler and render a conversation partial
    resp = await chat(session_id=session_id, message=message, role=role, image=image)
    if not resp:
        return templates.TemplateResponse(
            "partials/conversation.html",
            {
                "request": request,
                "chat_history": [
                    {"role": "system", "agent": "system", "text": "Chat failed unexpectedly.", "is_final": False}
                ],
                "delegation_chain": [],
                "run_id": None,
                "final_agent": None,
                "citations": [],
            },
        )
    # Cache last run info/trace in session for side panel rendering
    sid = resp.get("session_id")
    if sid and sid in sessions:
        sessions[sid]["last_run_info"] = resp.get("run_info")
        sessions[sid]["last_run_trace"] = resp.get("run_trace")
        sessions[sid]["last_delegation_chain"] = resp.get("delegation_chain")
        sessions[sid]["last_final_agent"] = resp.get("final_agent")
        sessions[sid]["last_seen"] = time.time()
    return templates.TemplateResponse(
        "partials/conversation.html",
        {
            "request": request,
            "chat_history": resp.get("chat_history", []),
            "delegation_chain": resp.get("delegation_chain", []),
            "run_id": resp.get("run_id"),
            "run_info": resp.get("run_info"),
            "run_trace": resp.get("run_trace"),
            "final_agent": resp.get("final_agent"),
            "citations": resp.get("citations", []),
            "session_id": resp.get("session_id"),
        },
    )


@app.get("/ui/events", response_class=HTMLResponse)
async def ui_events(request: Request, session_id: str):
    ev = await get_events(session_id=session_id)
    return templates.TemplateResponse(
        "partials/events.html",
        {"request": request, "run_events": ev.get("run_events", []), "run_complete": ev.get("run_complete", False)},
    )


@app.get("/ui/run_details", response_class=HTMLResponse)
async def ui_run_details(request: Request, session_id: str):
    sess = sessions.get(session_id, {})
    run_info = sess.get("last_run_info")
    run_trace = sess.get("last_run_trace")
    delegation_chain = sess.get("last_delegation_chain") or []
    final_agent = sess.get("last_final_agent")
    return templates.TemplateResponse(
        "partials/run_details.html",
        {
            "request": request,
            "run_info": run_info,
            "run_trace": run_trace or [],
            "delegation_chain": delegation_chain,
            "final_agent": final_agent,
            "session_id": session_id,
        },
    )


@app.post("/ui/reset", response_class=HTMLResponse)
async def ui_reset(
    request: Request,
    session_id: str = Form(None),
    new_thread: bool = Form(False),
):
    _ = await reset_agents(session_id=session_id, new_thread=new_thread)
    # Return an empty conversation; client can keep session id
    return templates.TemplateResponse(
        "partials/conversation.html",
        {"request": request, "chat_history": [], "delegation_chain": [], "run_id": None, "final_agent": None, "citations": []},
    )


@app.get("/events_sse")
async def events_sse(session_id: str):
    """Optional Server-Sent Events endpoint for smoother live updates."""
    async def event_generator():
        last_len = 0
        while True:
            sess = sessions.get(session_id, {})
            evs = sess.get("run_events", [])
            if len(evs) > last_len:
                for e in evs[last_len:]:
                    payload = json.dumps(e)
                    yield f"data: {payload}\n\n"
                last_len = len(evs)
            if sess.get("run_complete"):
                yield "event: done\n\n"
                break
            await asyncio.sleep(1)
    import asyncio  # local import to avoid unused in sync paths
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ----------------------------- Diagnostics & Maintenance ----------------------

def _session_reaper():
    while True:
        try:
            now = time.time()
            to_del = []
            for sid, sess in list(sessions.items()):
                last = sess.get("last_seen", now)
                if now - last > SESS_TTL_SECS:
                    to_del.append(sid)
            for sid in to_del:
                try:
                    del sessions[sid]
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(60)


@app.on_event("startup")
async def _on_startup():
    # Configure Azure Monitor if connection string is provided
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        conn = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        if conn:
            configure_azure_monitor(connection_string=conn)
    except Exception:
        pass
    # Start session reaper
    try:
        th = threading.Thread(target=_session_reaper, daemon=True)
        th.start()
    except Exception:
        pass


@app.get("/ui/diagnostics", response_class=HTMLResponse)
async def ui_diagnostics(request: Request, session_id: str):
    # Ensure we reflect the latest .env values in diagnostics
    try:
        load_dotenv(override=True)
    except Exception:
        pass
    sid = ensure_session(session_id)
    report = {
        "leader_id_env": os.getenv("LEADER_AGENT_ID"),
        "workers_env": {
            "librarian_rag": os.getenv("AGENT_ID_LIBRARIAN_RAG"),
            "therapy_expert": os.getenv("AGENT_ID_THERAPY_EXPERT"),
            "robotics_expert": os.getenv("AGENT_ID_ROBOTICS_EXPERT"),
        },
        "leader_tools": [],
        "mismatches": [],
    }
    try:
        with AIProjectClient(endpoint=ENDPOINT, credential=credential) as pc:
            client = pc.agents
            leader = sessions.get(sid, {}).get("leader")
            leader_id = getattr(leader, "id", None)
            found = None
            try:
                pager = client.list_agents()
                for page in getattr(pager, "by_page", lambda: [pager])():
                    for a in page:
                        if a.id == leader_id:
                            found = a
                            break
                    if found:
                        break
            except Exception:
                # fallback: iterate directly
                try:
                    for a in client.list_agents():
                        if a.id == leader_id:
                            found = a
                            break
                except Exception:
                    found = None
            if found:
                tools = getattr(found, "tools", []) or []
                for t in tools:
                    try:
                        if getattr(t, "type", None) == "connected_agent":
                            ca = getattr(t, "connected_agent", None) or {}
                            report["leader_tools"].append({
                                "name": getattr(ca, "name", None),
                                "id": getattr(ca, "id", None),
                            })
                    except Exception:
                        continue
                # Compare to env
                for name, env_id in report["workers_env"].items():
                    if not env_id:
                        report["mismatches"].append({"name": name, "reason": "env missing"})
                        continue
                    match = any(item.get("id") == env_id for item in report["leader_tools"])
                    if not match:
                        report["mismatches"].append({"name": name, "reason": "leader not wired to this id"})
    except Exception as e:
        report["error"] = str(e)

    return templates.TemplateResponse(
        "partials/diagnostics.html",
        {"request": request, "session_id": sid, "report": report},
    )
