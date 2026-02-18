from langchain_core.documents import Document
from langchain.agents.middleware import AgentState

class StateWithDocuments(AgentState):
    context: list[Document]