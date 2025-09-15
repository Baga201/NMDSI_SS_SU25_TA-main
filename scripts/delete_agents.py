"""
Safely delete all existing agents in the Azure AI Agents project.
- Snapshot IDs first to avoid mutating the list while iterating.
- Then delete each agent with optional retry logic.
"""
import os
import time
from typing import List
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError

load_dotenv(override=True)

ENDPOINT = os.getenv("PROJECT_ENDPOINT")
if not ENDPOINT:
    raise SystemExit("PROJECT_ENDPOINT is not set. Check your .env")

credential = DefaultAzureCredential()

def snapshot_agent_ids(agents_client) -> List[str]:
    """Collect all agent IDs before deletion (no mutation during paging)."""
    ids = []
    pager = agents_client.list_agents()
    for page in pager.by_page():
        for agent in page:
            ids.append(agent.id)
    return ids

def delete_with_retries(agents_client, agent_id: str, max_retries: int = 3) -> None:
    """Delete an agent with simple retry/backoff for transient errors."""
    delay = 0.8
    for attempt in range(1, max_retries + 1):
        try:
            agents_client.delete_agent(agent_id)
            print(f"Deleted: {agent_id}")
            return
        except ResourceNotFoundError:
            # Already deleted, fine
            print(f"Not found (already deleted): {agent_id}")
            return
        except HttpResponseError as e:
            status = getattr(e, "status_code", None)
            if status in (409, 429) or (status and status >= 500):
                # Transient; back off and retry
                print(f"Transient error {status} on {agent_id} (attempt {attempt}/{max_retries}); retrying…")
                time.sleep(delay)
                delay *= 1.6
                continue
            raise
        except Exception as e:
            print(f"Unexpected error deleting {agent_id}: {e}")
            raise

if __name__ == "__main__":
    with AIProjectClient(endpoint=ENDPOINT, credential=credential) as client:
        agents_client = client.agents

        print("Taking snapshot of existing agents…")
        ids = snapshot_agent_ids(agents_client)
        print(f"Found {len(ids)} agents.")

        # Optional: reverse to delete newest first
        for agent_id in reversed(ids):
            delete_with_retries(agents_client, agent_id)
