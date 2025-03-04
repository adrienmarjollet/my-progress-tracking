from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    @abstractmethod
    def get_response(self, question):
        """Get a response from the LLM for the given question"""
        pass
    
    @abstractmethod
    def classify_theme(self, question, categories):
        """Classify the question into one of the given categories"""
        pass
