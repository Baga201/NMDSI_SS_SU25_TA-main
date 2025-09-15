# "ChatMode" for project-specific multi-agent system leveraging VS Code MCP and Copilot
# Author: GitHub Copilot
#
# This class provides enhanced system instructions and context for agents, integrating VS Code MCP and Copilot features.
#
import os
from typing import List, Dict, Any

class ChatMode:
	"""
	Custom chat mode for the NMDSI_SS_SU25_TA multi-agent system.
	Provides project-specific system instructions and context, leveraging VS Code MCP and Copilot.
	"""
	def __init__(self, agents: List[str], vector_store_id: str = None):
		self.agents = agents
		self.vector_store_id = vector_store_id or os.getenv("VECTOR_STORE_ID", "")
		self.project_endpoint = os.getenv("PROJECT_ENDPOINT", "")
		self.model_deployment_name = os.getenv("MODEL_DEPLOYMENT_NAME", "")

	def get_system_instructions(self) -> str:
		"""
		Returns enhanced system instructions for the multi-agent team, including citation and human-in-the-loop policy.
		"""
		instructions = f"""
You are a multi-agent system for research and therapy, coordinated by a leader agent. Your team includes:
- Therapist (PT expert)
- Robotics specialist
- Librarian/RAG (PDF search)
- Student/Patient channels

Always cite answers from PDFs using inline bracket notation. If no citation exists, prompt for human-expert input and record the question for review.
Use Azure AI Agents File Search and VS Code MCP tools for file search, code interpretation, and OpenAPI access.
Environment variables:
- PROJECT_ENDPOINT: {self.project_endpoint}
- MODEL_DEPLOYMENT_NAME: {self.model_deployment_name}
- VECTOR_STORE_ID: {self.vector_store_id}

Follow PEP 8, add docstrings, and ensure error handling. Do not modify files under ./data.
"""
		return instructions

	def get_agent_context(self) -> Dict[str, Any]:
		"""
		Returns context for agent instantiation, including model and vector store info.
		"""
		return {
			"agents": self.agents,
			"vector_store_id": self.vector_store_id,
			"project_endpoint": self.project_endpoint,
			"model_deployment_name": self.model_deployment_name,
		}

	def get_tools(self) -> List[str]:
		"""
		Returns a list of enabled VS Code MCP tools for the agents.
		"""
		return [
			"File Search",
			"Code Interpreter",
			"OpenAPI",
			"Azure AI Agents",
			"ConnectedAgentTool",
		]

	def enhance_instructions(self, custom: str) -> str:
		"""
		Allows further customization of system instructions.
		"""
		return self.get_system_instructions() + "\n" + custom
