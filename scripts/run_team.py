"""
CLI sanity runner for your config-driven multi-agent team.

This script reads a YAML configuration (matching the format you provided) and
instantiates workers and a leader agent using the Azure AI Agents API. It can
optionally delete any existing agents in the project, then create new ones.
After creating the agents, it spins up a new thread, posts a sample user
question, and invokes the leader via `create_and_process`. It then prints
messages from the thread to the console, inserting file citations when
appropriate and highlighting structured results from worker JSON.

Usage:
  python run_team.py [--delete-old-agents] [--question "your query"]
    python run_team.py --update-existing [--question "your query"]

Environment variables:
  PROJECT_ENDPOINT   – Azure AI Foundry project endpoint (required)
  VECTOR_STORE_ID    – Vector store ID for librarian_rag (required if librarian is enabled)
  TEAM_CONFIG_PATH   – Path to team.yaml (default: configs/team.yaml)

Note: This script is designed for demonstration and local testing. For a
production or web environment, see backend.py.
"""

import os
import json
import yaml
import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import (
    MessageRole,
    ListSortOrder,
    FileSearchTool,
    ConnectedAgentTool,
)

load_dotenv(override=True)

# Environment variables
CONFIG_PATH = os.getenv("TEAM_CONFIG_PATH", "configs/team.yaml")
ENDPOINT = os.getenv("PROJECT_ENDPOINT")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")
credential = DefaultAzureCredential()


def load_team_config(path: str) -> Dict[str, Any]:
    """
    Load the YAML team configuration and substitute ${VECTOR_STORE_ID} if defined.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    team_cfg = cfg.get("team", {})
    if "workers" not in team_cfg and "workers" in cfg:
        team_cfg["workers"] = cfg["workers"]
    # Substitute VECTOR_STORE_ID if present
    if VECTOR_STORE_ID:
        for w in team_cfg.get("workers", []):
            for t in w.get("tools", []):
                if isinstance(t, dict) and "vector_store_ids" in t:
                    t["vector_store_ids"] = [
                        VECTOR_STORE_ID if str(v).strip() == "${VECTOR_STORE_ID}" else v
                        for v in t["vector_store_ids"]
                    ]
    return team_cfg


def update_worker_in_place(client, agent_id: str, w_cfg: Dict[str, Any]) -> bool:
    """Update a worker agent in place: instructions, model, and tools/resources if needed.
    Returns True on success, False otherwise.
    """
    try:
        # Build tool definitions/resources for librarian_rag if applicable
        tools = None
        tool_resources = None
        if w_cfg.get("name") == "librarian_rag":
            vsids: List[str] = []
            for t in w_cfg.get("tools", []):
                if isinstance(t, dict) and t.get("type") == "file_search":
                    vsids = t.get("vector_store_ids") or []
                    break
            if not vsids and VECTOR_STORE_ID:
                vsids = [VECTOR_STORE_ID]
            if vsids:
                fs_tool = FileSearchTool(vector_store_ids=vsids)
                tools = fs_tool.definitions
                tool_resources = fs_tool.resources

        # Attempt SDK update
        try:
            client.update_agent(
                agent_id,
                model=w_cfg.get("model"),
                name=w_cfg.get("name"),
                instructions=w_cfg.get("instructions"),
                tools=tools,
                tool_resources=tool_resources,
            )
            print(f"[update] worker {w_cfg.get('name')} ({agent_id})")
            return True
        except AttributeError:
            # Older SDKs may expose a named parameter signature
            client.update_agent(
                agent_id=agent_id,
                model=w_cfg.get("model"),
                name=w_cfg.get("name"),
                instructions=w_cfg.get("instructions"),
                tools=tools,
                tool_resources=tool_resources,
            )
            print(f"[update] worker {w_cfg.get('name')} ({agent_id})")
            return True
    except Exception as e:
        print(f"[warn] update worker {w_cfg.get('name')} failed: {e}")
        return False


def update_leader_in_place(client, leader_id: str, team_cfg: Dict[str, Any], worker_ids: Dict[str, str]) -> bool:
    """Update leader agent in place: instructions, model, and connected-agent tools.
    Returns True on success, False otherwise.
    """
    try:
        leader_tools: List[Any] = []
        # Rebuild connected-agent tool definitions using current worker IDs
        for w in team_cfg.get("workers", []):
            wname = w.get("name")
            wid = worker_ids.get(wname)
            if not wid:
                print(f"[warn] no env id for worker {wname}; skipping tool wiring")
                continue
            cat = ConnectedAgentTool(id=wid, name=wname, description=f"Delegated agent: {wname}")
            leader_tools.append(cat.definitions[0])

        try:
            client.update_agent(
                leader_id,
                model=team_cfg.get("leader_model"),
                name=team_cfg.get("leader_name"),
                instructions=team_cfg.get("leader_instructions"),
                tools=leader_tools,
            )
            print(f"[update] leader {team_cfg.get('leader_name')} ({leader_id})")
            return True
        except AttributeError:
            client.update_agent(
                agent_id=leader_id,
                model=team_cfg.get("leader_model"),
                name=team_cfg.get("leader_name"),
                instructions=team_cfg.get("leader_instructions"),
                tools=leader_tools,
            )
            print(f"[update] leader {team_cfg.get('leader_name')} ({leader_id})")
            return True
    except Exception as e:
        print(f"[warn] update leader failed: {e}")
        return False


def create_workers(client, team_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create worker agents defined in the configuration. The librarian gets a
    FileSearchTool if configured with vector_store_ids or if VECTOR_STORE_ID is set.
    """
    workers: Dict[str, Any] = {}
    for w in team_cfg.get("workers", []):
        name = w["name"]
        if name == "librarian_rag":
            vsids: List[str] = []
            for t in w.get("tools", []):
                if isinstance(t, dict) and t.get("type") == "file_search":
                    vsids = t.get("vector_store_ids") or []
                    break
            # If config didn't specify, fall back to env
            if not vsids and VECTOR_STORE_ID:
                vsids = [VECTOR_STORE_ID]
            if not vsids:
                raise ValueError("VECTOR_STORE_ID missing for librarian_rag")
            fs_tool = FileSearchTool(vector_store_ids=vsids)
            agent = client.create_agent(
                model=w["model"],
                name=name,
                instructions=w["instructions"],
                tools=fs_tool.definitions,
                tool_resources=fs_tool.resources,
            )
            print(f"[create] {name} (FileSearch: {vsids})")
        else:
            agent = client.create_agent(
                model=w["model"], name=name, instructions=w["instructions"]
            )
            print(f"[create] {name}")
        workers[name] = agent
    return workers


def create_leader(client, team_cfg: Dict[str, Any], workers: Dict[str, Any]):
    """
    Create the leader agent and attach each worker as a ConnectedAgentTool.
    """
    leader_tools: List[Any] = []
    for n, a in workers.items():
        cat = ConnectedAgentTool(id=a.id, name=n, description=f"Delegated agent: {n}")
        leader_tools.append(cat.definitions[0])
    leader = client.create_agent(
        model=team_cfg["leader_model"],
        name=team_cfg["leader_name"],
        instructions=team_cfg["leader_instructions"],
        tools=leader_tools,
    )
    print(f"[create] leader: {team_cfg['leader_name']}")
    return leader


def print_messages(client, thread_id: str) -> None:
    """
    Print all messages in the thread. If a message contains annotations with
    file citations, insert markers; if it is valid JSON with a 'result' key,
    highlight that as a structured response.
    """
    msgs = client.messages.list(thread_id=thread_id, order=ListSortOrder.ASCENDING)
    for m in msgs:
        who = getattr(m, "agent_name", None) or getattr(m, "sender", None) or m.role
        if not m.text_messages:
            continue
        last = m.text_messages[-1]
        text = last.text.value
        # Insert citation markers
        anns = getattr(last, "annotations", None)
        if anns:
            for a in anns:
                if hasattr(a, "file_citation"):
                    fid = a.file_citation.file_id
                    try:
                        text = text.replace(a.text, f" [{fid}]")
                    except Exception:
                        pass
        # Show structured result if JSON
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "result" in obj:
                print(f"{who}: (structured) result={obj.get('result')!r}")
                continue
        except Exception:
            pass
        print(f"{who}: {text}")


def main():
    parser = argparse.ArgumentParser(description="Run multi-agent team sample")
    parser.add_argument("--delete-old-agents", action="store_true",
                        help="Delete all existing agents before creating new ones")
    parser.add_argument("--question", default="Outline early post-stroke gait rehab options and cite sources.")
    parser.add_argument("--update-existing", action="store_true",
                        help="Update existing agents in place using env-provided IDs and team.yaml")
    parser.add_argument("--write-env-updates", action="store_true",
                        help="After creating agents, write their IDs back into the .env file (safe backup)")
    args = parser.parse_args()

    if not ENDPOINT:
        raise SystemExit("PROJECT_ENDPOINT is required")

    team_cfg = load_team_config(CONFIG_PATH)
    with AIProjectClient(endpoint=ENDPOINT, credential=credential) as pc:
        client = pc.agents

        if args.update_existing:
            print("[update-existing] applying team.yaml to existing agents …")
            # Gather env-provided IDs
            leader_id = os.getenv("LEADER_AGENT_ID")
            wid_map = {
                "librarian_rag": os.getenv("AGENT_ID_LIBRARIAN_RAG"),
                "therapy_expert": os.getenv("AGENT_ID_THERAPY_EXPERT"),
                "robotics_expert": os.getenv("AGENT_ID_ROBOTICS_EXPERT"),
            }
            if not leader_id:
                raise SystemExit("LEADER_AGENT_ID is required for --update-existing")

            # Update workers
            ok_workers = True
            for w in team_cfg.get("workers", []):
                wname = w.get("name")
                wid = wid_map.get(wname)
                if not wid:
                    print(f"[warn] missing env id for worker {wname}; skipping")
                    ok_workers = False
                    continue
                _ = update_worker_in_place(client, wid, w)

            # Update leader with connected-agent tools rebuilt
            ok_leader = update_leader_in_place(client, leader_id, team_cfg, wid_map)

            print("[update-existing] done.")
            # Optionally skip running a question in update mode; exit here
            return

        # Optional cleanup: snapshot IDs first to avoid mutation during paging
        if args.delete_old_agents:
            print("[cleanup] deleting existing agents …")
            try:
                ids = []
                pager = client.list_agents()
                # Take a snapshot of agent IDs
                try:
                    for page in pager.by_page():
                        for a in page:
                            ids.append(a.id)
                except Exception:
                    # Fallback: iterate directly if by_page isn't supported
                    for a in client.list_agents():
                        ids.append(a.id)
                for aid in reversed(ids):
                    try:
                        client.delete_agent(aid)
                        print(f"  deleted {aid}")
                    except Exception as e:
                        print(f"  skip delete ({e})")
            except Exception as e:
                print(f"[cleanup] listing failed: {e}")

        # Create workers and leader
        workers = create_workers(client, team_cfg)
        leader = create_leader(client, team_cfg, workers)

        # Optionally write IDs back to .env to keep future runs stable
        if args.write_env_updates or os.getenv("WRITE_ENV_ON_CREATE") == "1":
            try:
                env_path = Path(".env")
                updates = {"LEADER_AGENT_ID": getattr(leader, "id", None)}
                for wname, agent in workers.items():
                    key = f"AGENT_ID_{str(wname).upper()}"
                    updates[key] = getattr(agent, "id", None)

                # Safe update: backup and preserve comments/unknown lines
                def _apply_env_updates(path: Path, pairs: Dict[str, str]):
                    original = path.read_text(encoding="utf-8") if path.exists() else ""
                    backup = path.with_suffix(path.suffix + ".bak")
                    if original:
                        try:
                            shutil.copyfile(path, backup)
                            print(f"[env] backup written: {backup}")
                        except Exception as e:
                            print(f"[env] backup skipped: {e}")
                    lines = original.splitlines() if original else []
                    keys_lower = {k.lower(): k for k in pairs.keys() if pairs[k]}
                    seen = set()
                    out = []
                    for line in lines:
                        stripped = line.strip()
                        if not stripped or stripped.startswith("#"):
                            out.append(line)
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            kl = k.strip().lower()
                            if kl in keys_lower:
                                real_key = keys_lower[kl]
                                out.append(f"{real_key}={pairs[real_key]}")
                                seen.add(kl)
                            else:
                                out.append(line)
                        else:
                            out.append(line)
                    # append any missing keys
                    for kl, real_key in keys_lower.items():
                        if kl not in seen and pairs[real_key]:
                            out.append(f"{real_key}={pairs[real_key]}")
                    new_text = "\n".join(out) + "\n"
                    path.write_text(new_text, encoding="utf-8")
                    print(f"[env] updated: {path}")

                _apply_env_updates(env_path, updates)
                print("[env] You can now reuse these IDs via LEADER_AGENT_ID and AGENT_ID_* to avoid recreation.")
            except Exception as e:
                print(f"[env] update failed: {e}")

        # Create a thread and post a user question
        thread = client.threads.create()
        client.messages.create(thread_id=thread.id, role=MessageRole.USER, content=args.question)

        print("[run] leader.create_and_process …")
        try:
            run = client.runs.create_and_process(thread_id=thread.id, agent_id=leader.id)
        except Exception:
            run = client.runs.create(thread_id=thread.id, agent_id=leader.id)
            # if your SDK exposes a process method, you can start processing here
            # try:
            #     client.runs.process(thread_id=thread.id, run_id=run.id)
            # except Exception:
            #     pass

        print(f"[run] status: {getattr(run, 'status', 'unknown')}")
        # simple bounded poll when not using create_and_process
        try:
            import time as _t
            start_ts = _t.time()
            while getattr(run, 'status', '') not in ["completed", "failed", "cancelled"] and _t.time() - start_ts < 120:
                _t.sleep(0.5)
                run = client.runs.get(thread_id=thread.id, run_id=run.id)
        except Exception:
            pass
        if getattr(run, "status", "") == "failed":
            print(f"[run] error: {getattr(run, 'last_error', None)}")

        # Print all messages with citations / structured result
        print_messages(client, thread.id)
        print("\nTip: If you don’t see librarian citations, check VECTOR_STORE_ID and ensure your PDFs were ingested.\n")


if __name__ == "__main__":
    main()
