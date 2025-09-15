"""
Agent builder utilities for Azure AI Foundry multi-agent system.
"""
from azure_ai_agents import AgentsClient

def build_agent(config, endpoint, model_name):
    """Instantiate agent from config dict."""
    client = AgentsClient(endpoint=endpoint, model_deployment_name=model_name)
    # Convert tool dicts to SDK tool definitions if present
    config = dict(config)  # shallow copy
    if 'tools' in config and config['tools']:
        sdk_tools = []
        for tool in config['tools']:
            if isinstance(tool, dict) and tool.get('type') == 'file_search':
                from azure_ai_agents import FileSearchTool
                vector_store_ids = tool.get('vector_store_ids', [])
                file_search_tool = FileSearchTool(vector_store_ids=vector_store_ids)
                sdk_tools.extend(file_search_tool.definitions)
            elif not isinstance(tool, dict):
                sdk_tools.append(tool)
        config['tools'] = sdk_tools
    return client.create_agent(config)
