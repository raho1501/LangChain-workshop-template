from typing import Any
from langchain_community.vectorstores import FAISS
from langchain.agents.middleware import AgentMiddleware, AgentState
from .StateWithDocuments import StateWithDocuments
import warnings

class RetrieveDocumentsMiddleware(AgentMiddleware[StateWithDocuments]):
    state_schema = StateWithDocuments

    def __init__(self, vector_store : FAISS, similarity_score_threshold : float):
        self.vector_store = vector_store
        self.similarity_score_threshold = similarity_score_threshold
        super().__init__()



    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            retrieved_docs = self.vector_store.similarity_search_with_relevance_scores(last_message.text, k=20)

        for doc, score in retrieved_docs:
            doc.metadata["relevance_score"] = score

        docs_content = "".join((f"Source: {doc.metadata['source']}\nContent: {doc.page_content}") for doc, score in retrieved_docs if score > self.similarity_score_threshold)
        
        if len(docs_content) == 0:
            augmented_message_content = "Ignore previous instruction and reply to the user with the text \"Sorry, I can\'t answer that\""
            return {
                "messages": [last_message.model_copy(update={"content": augmented_message_content})],
            }
        else:
            augmented_message_content = (
                f"{last_message.text}\n\n"
                "Use the following context to answer the query and add the used source at the end:\n"
                f"\n\n{docs_content}"
            )

        return {
            "messages": [last_message.model_copy(update={"content": augmented_message_content})],
            "context": retrieved_docs,
        }