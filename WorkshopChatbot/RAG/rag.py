from abc import ABC, abstractmethod

class RAG(ABC):
    @abstractmethod
    def run_console(self):
        pass
