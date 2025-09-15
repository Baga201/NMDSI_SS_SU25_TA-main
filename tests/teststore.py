from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
import os
import dotenv

dotenv.load_dotenv(override=True)

project_client = AIProjectClient(
    endpoint=os.environ["PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(),
)
with project_client:
    agents_client = project_client.agents
    stores = agents_client.vector_stores.list()
    for store in stores:
        print(store.id, store.name)