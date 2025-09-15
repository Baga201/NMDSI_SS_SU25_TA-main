"""
Unit tests for sender @ receiver labeling logic in scripts/backend.py.
These tests mock Azure SDK interactions and focus on the transcript synthesis.
"""
from types import SimpleNamespace
from typing import Any, Dict, List
import sys
import types

# We will monkeypatch AIProjectClient used inside backend.py so importing backend does not hit Azure
class _DummyRuns:
    def create(self, *a, **k):
        return SimpleNamespace(id="run_1", status="completed", steps=[])
    def get(self, *a, **k):
        return SimpleNamespace(id="run_1", status="completed", steps=[])

class _DummyMessages:
    def __init__(self, messages):
        self._messages = messages
    def create(self, *a, **k):
        return None
    def list(self, thread_id: str, order: Any=None):
        return self._messages

class _DummyThreads:
    def create(self):
        return SimpleNamespace(id="thread_1")

class _DummyAgents:
    def __init__(self, messages):
        self.messages = _DummyMessages(messages)
        self.runs = _DummyRuns()
        self.threads = _DummyThreads()
    def create_agent(self, *a, **k):
        return SimpleNamespace(id="asst_leader", name=k.get("name","leader"))
    def delete_agent(self, *a, **k):
        return None
    def list_agents(self):
        return []
    def update_agent(self, *a, **k):
        return None

class _DummyPC:
    def __init__(self, messages):
        self.agents = _DummyAgents(messages)
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False

# Fabricate minimal message objects used by backend
class _MsgText:
    def __init__(self, value: str, annotations: List[Any]=None):
        self.value = value
        self.annotations = annotations or []

class _Msg:
    def __init__(self, role: str, text: str, agent_id: str=None, agent_name: str=None, created_at: str="t0"):
        self.role = role
        self.text_messages = [SimpleNamespace(text=_MsgText(text))]
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.sender = agent_name
        self.created_at = created_at


def _install_azure_stubs(messages: List[Any]):
    """Install stub Azure modules in sys.modules before importing backend."""
    # Root packages
    azure_mod = types.ModuleType("azure")
    ai_mod = types.ModuleType("azure.ai")
    identity_mod = types.ModuleType("azure.identity")
    projects_mod = types.ModuleType("azure.ai.projects")
    agents_mod = types.ModuleType("azure.ai.agents")
    models_mod = types.ModuleType("azure.ai.agents.models")

    class DefaultAzureCredential:
        def __init__(self, *a, **k):
            pass

    identity_mod.DefaultAzureCredential = DefaultAzureCredential

    class MessageRole:
        USER = "user"
        AGENT = "agent"
        SYSTEM = "system"

    class ListSortOrder:
        ASCENDING = "ASC"
        DESCENDING = "DESC"

    class FileSearchTool:
        def __init__(self, vector_store_ids=None):
            self.definitions = [
                {"type": "file_search", "name": "file_search"}
            ]
            self.resources = {"file_search": {"vector_store_ids": vector_store_ids or []}}

    class ConnectedAgentTool:
        def __init__(self, id: str, name: str, description: str = ""):
            self.definitions = [
                {"type": "connected_agent", "connected_agent": {"id": id, "name": name, "description": description}}
            ]

    class MessageInputTextBlock:
        def __init__(self, text: str):
            self.text = text

    class MessageInputImageFileBlock:
        def __init__(self, image_file: Any):
            self.image_file = image_file

    class MessageImageFileParam:
        def __init__(self, file_id: str, detail: str = "high"):
            self.file_id = file_id
            self.detail = detail

    class FilePurpose:
        AGENTS = "agents"

    models_mod.MessageRole = MessageRole
    models_mod.ListSortOrder = ListSortOrder
    models_mod.FileSearchTool = FileSearchTool
    models_mod.ConnectedAgentTool = ConnectedAgentTool
    models_mod.MessageInputTextBlock = MessageInputTextBlock
    models_mod.MessageInputImageFileBlock = MessageInputImageFileBlock
    models_mod.MessageImageFileParam = MessageImageFileParam
    models_mod.FilePurpose = FilePurpose

    # AIProjectClient stub (unused once we patch backend.AIProjectClient, but needed for import)
    class AIProjectClient:
        def __init__(self, *a, **k):
            self.agents = _DummyAgents([])
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    projects_mod.AIProjectClient = AIProjectClient

    # Register modules in sys.modules
    sys.modules.setdefault("azure", azure_mod)
    sys.modules.setdefault("azure.ai", ai_mod)
    sys.modules.setdefault("azure.identity", identity_mod)
    sys.modules.setdefault("azure.ai.projects", projects_mod)
    sys.modules.setdefault("azure.ai.agents", agents_mod)
    sys.modules.setdefault("azure.ai.agents.models", models_mod)

    return {
        "MessageRole": MessageRole,
        "ListSortOrder": ListSortOrder,
    }


def _build_backend_with_messages(messages: List[Any]):
    # Ensure azure stubs are present before import
    _install_azure_stubs(messages)
    import scripts.backend as backend
    # Monkeypatch AIProjectClient within backend namespace to drive behavior
    backend.AIProjectClient = lambda *a, **k: _DummyPC(messages)
    # Provide a fake credential and endpoint
    backend.ENDPOINT = "test_endpoint"
    backend.credential = object()
    return backend


def test_sender_receiver_labeling_simple_flow(monkeypatch):
    # Given: one user message, one leader reply (final)
    msgs = [
        _Msg(role="user", text="Hello", agent_id=None, agent_name=None, created_at="t1"),
        _Msg(role="agent", text="Leader reply", agent_id="asst_leader", agent_name="orchestrator", created_at="t2"),
    ]
    backend = _build_backend_with_messages(msgs)

    # Session and team config
    sid = backend.ensure_session(None)
    team_cfg = {"leader_name": "orchestrator", "leader_model": "gpt-4o", "leader_instructions": "lead" , "workers": []}
    backend.build_agents_for_session(sid, team_cfg)

    # Call chat
    import asyncio
    resp = asyncio.get_event_loop().run_until_complete(
        backend.chat(session_id=sid, message="Hello", role="Student", image=None)
    )

    ch = resp["chat_history"]
    # Expect: first bubble user @ orchestrator, last bubble orchestrator @ user (final)
    assert any(e.get("role") == "user" and e.get("delegated_to") == "orchestrator" for e in ch)
    finals = [e for e in ch if e.get("is_final")]
    assert len(finals) >= 1
    assert finals[-1].get("agent") == "orchestrator"
    assert finals[-1].get("delegated_to") == "user"


def test_leader_worker_dialogue_insertion(monkeypatch):
    # Given: user, leader delegates to librarian (tool step), worker replies
    # Simulate run steps via run_trace by patching after chat returns using a controlled response
    worker_id = "asst_worker_1"
    msgs = [
        _Msg(role="user", text="Find sources", created_at="t1"),
        _Msg(role="agent", text="Working on it", agent_id="asst_leader", agent_name="orchestrator", created_at="t2"),
    ]
    backend = _build_backend_with_messages(msgs)

    sid = backend.ensure_session(None)
    team_cfg = {
        "leader_name": "orchestrator",
        "leader_model": "gpt-4o",
        "leader_instructions": "lead",
        "workers": [{"name": "librarian_rag", "model": "gpt-4o-mini", "instructions": "cite", "tools": [{"type":"file_search", "vector_store_ids":["vs1"]}]}],
    }
    backend.build_agents_for_session(sid, team_cfg)

    # Patch sessions to include worker id mapping
    backend.sessions[sid]["agents"]["librarian_rag"].id = worker_id

    # Monkeypatch _stream_run_events to NOOP and patch run retrieval to include steps
    def _fake_stream(*a, **k):
        pass
    backend._stream_run_events = _fake_stream

    # Patch client.runs.get to return a synthetic run with steps for delegation and reply
    class _ToolCall:
        def __init__(self, name, input, output=None):
            self.name = name
            self.tool_name = name
            self.input = input
            self.output = output

    class _Step:
        def __init__(self, agent_id, tool_name, tool_input, tool_output, text=None):
            self.id = f"step_{tool_name}"
            self.step_type = "tool_call"
            self.agent_id = agent_id
            self.tool_call = _ToolCall(tool_name, tool_input, tool_output)
            self.status = "completed"
            self.timestamp = "t3"
            self.text = text

    fake_run = SimpleNamespace(id="run_1", status="completed", steps=[
        _Step(agent_id=worker_id, tool_name="librarian_rag", tool_input={"prompt":"find sources"}, tool_output={"result":"ok"}),
    ])

    class _DummyRuns2(_DummyRuns):
        def get(self, *a, **k):
            return fake_run
        def create(self, *a, **k):
            return SimpleNamespace(id="run_1", status="running", steps=[])

    backend.AIProjectClient = lambda *a, **k: SimpleNamespace(
        __enter__=lambda self: self,
        __exit__=lambda *args: False,
        agents=SimpleNamespace(
            messages=_DummyMessages(msgs),
            runs=_DummyRuns2(),
            threads=_DummyThreads(),
            create_agent=lambda **kw: SimpleNamespace(id="asst_leader", name=kw.get("name","leader")),
        )
    )

    import asyncio
    resp = asyncio.get_event_loop().run_until_complete(
        backend.chat(session_id=sid, message="Find sources", role="Student", image=None)
    )
    ch = resp["chat_history"]
    # We expect leader->worker delegation and worker->leader reply bubbles inserted
    has_delegate = any(e.get("agent") == "orchestrator" and e.get("delegated_to") == "librarian_rag" for e in ch)
    has_reply = any(e.get("agent") == "librarian_rag" and e.get("delegated_from") == "orchestrator" for e in ch)
    assert has_delegate and has_reply
