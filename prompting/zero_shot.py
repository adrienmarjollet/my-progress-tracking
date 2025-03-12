from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable
import json

class PromptTemplate:
    """Class to handle prompt templates for zero-shot prompting."""
    
    def __init__(self, template: str):
        """
        Initialize with a template string containing placeholders.
        
        Args:
            template: String with {placeholder} format for variable substitution
        """
        self.template = template
    
    def format(self, **kwargs) -> str:
        """
        Format the template with provided values.
        
        Args:
            **kwargs: Key-value pairs to substitute in the template
            
        Returns:
            Formatted prompt string
        """
        return self.template.format(**kwargs)
    
    @classmethod
    def from_file(cls, file_path: str) -> 'PromptTemplate':
        """
        Load a template from a file.
        
        Args:
            file_path: Path to the template file
            
        Returns:
            PromptTemplate instance
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            template = f.read()
        return cls(template)


class ZeroShotTask(ABC):
    """Abstract base class for zero-shot tasks."""
    
    @abstractmethod
    def create_prompt(self, **kwargs) -> str:
        """Create a prompt for this task."""
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> Any:
        """Parse the model's response for this task."""
        pass


class ZeroShotClassification(ZeroShotTask):
    """Class for zero-shot classification tasks."""
    
    def __init__(self, 
                 labels: List[str], 
                 prompt_template: Optional[PromptTemplate] = None,
                 multi_label: bool = False):
        """
        Initialize zero-shot classification.
        
        Args:
            labels: List of possible classification labels
            prompt_template: Custom prompt template (optional)
            multi_label: Whether multiple labels can be assigned (True) or just one (False)
        """
        self.labels = labels
        self.multi_label = multi_label
        
        # Default template if none provided
        default_template = (
            "Classify the following text into {label_type}.\n\n"
            "Text: {text}\n\n"
            "Possible labels: {labels}\n\n"
            "The classification should be exactly one of the provided labels. "
            "Respond with only the label and nothing else."
        )
        
        self.prompt_template = prompt_template or PromptTemplate(default_template)
    
    def create_prompt(self, text: str, label_type: str = "categories") -> str:
        """
        Create a classification prompt.
        
        Args:
            text: Text to classify
            label_type: Description of what the labels represent (e.g., "categories", "emotions")
            
        Returns:
            Formatted prompt string
        """
        return self.prompt_template.format(
            text=text,
            labels=", ".join(self.labels),
            label_type=label_type
        )
    
    def parse_response(self, response: str) -> Union[str, List[str]]:
        """
        Parse classification response.
        
        Args:
            response: Model response string
            
        Returns:
            Single label or list of labels (if multi_label=True)
        """
        response = response.strip()
        
        if self.multi_label:
            # Try to parse as JSON list first
            try:
                labels = json.loads(response)
                if isinstance(labels, list):
                    return [label for label in labels if label in self.labels]
            except:
                pass
                
            # Fall back to simple string splitting
            potential_labels = [label.strip() for label in response.split(',')]
            return [label for label in potential_labels if label in self.labels]
        else:
            # For single label, find the first match
            for label in self.labels:
                if label.lower() in response.lower():
                    return label
            # If no exact match, return the response
            return response


class ZeroShotQA(ZeroShotTask):
    """Class for zero-shot question answering."""
    
    def __init__(self, prompt_template: Optional[PromptTemplate] = None):
        """
        Initialize zero-shot QA task.
        
        Args:
            prompt_template: Custom prompt template (optional)
        """
        # Default template if none provided
        default_template = (
            "Answer the following question based on the provided context.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        
        self.prompt_template = prompt_template or PromptTemplate(default_template)
    
    def create_prompt(self, question: str, context: Optional[str] = None) -> str:
        """
        Create a QA prompt.
        
        Args:
            question: Question to answer
            context: Optional context information
            
        Returns:
            Formatted prompt string
        """
        return self.prompt_template.format(
            question=question,
            context=context or "No additional context provided."
        )
    
    def parse_response(self, response: str) -> str:
        """
        Parse QA response.
        
        Args:
            response: Model response string
            
        Returns:
            Cleaned answer string
        """
        return response.strip()


class ZeroShotExtraction(ZeroShotTask):
    """Class for zero-shot information extraction tasks."""
    
    def __init__(self, 
                 schema: Dict[str, str],
                 output_format: str = "json",
                 prompt_template: Optional[PromptTemplate] = None):
        """
        Initialize zero-shot extraction.
        
        Args:
            schema: Dictionary of field names and their descriptions
            output_format: Format for output ("json" or "text")
            prompt_template: Custom prompt template (optional)
        """
        self.schema = schema
        self.output_format = output_format
        
        # Create schema description for the prompt
        schema_desc = "\n".join([f"- {field}: {desc}" for field, desc in schema.items()])
        
        # Default template if none provided
        default_template = (
            "Extract the following information from the text below.\n\n"
            "Text: {text}\n\n"
            "Information to extract:\n{schema_desc}\n\n"
            "Provide the extracted information in {output_format} format."
        )
        
        self.prompt_template = prompt_template or PromptTemplate(default_template)
    
    def create_prompt(self, text: str) -> str:
        """
        Create an extraction prompt.
        
        Args:
            text: Text to extract information from
            
        Returns:
            Formatted prompt string
        """
        schema_desc = "\n".join([f"- {field}: {desc}" for field, desc in self.schema.items()])
        
        return self.prompt_template.format(
            text=text,
            schema_desc=schema_desc,
            output_format=self.output_format
        )
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse extraction response.
        
        Args:
            response: Model response string
            
        Returns:
            Dictionary of extracted information
        """
        if self.output_format.lower() == "json":
            # Try to parse JSON from the response
            try:
                # Find JSON-like structure in the response
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > 0:
                    json_str = response[start_idx:end_idx]
                    extracted_data = json.loads(json_str)
                    # Filter to only include fields from our schema
                    return {k: v for k, v in extracted_data.items() if k in self.schema}
                
            except json.JSONDecodeError:
                pass
        
        # Fallback to simple text extraction
        result = {}
        for field in self.schema:
            field_lower = field.lower()
            search_patterns = [
                f"{field}:", f"{field_lower}:",
                f"\"{field}\":", f"\"{field_lower}\":",
                f"'{field}':", f"'{field_lower}':"
            ]
            
            for pattern in search_patterns:
                if pattern in response:
                    start_idx = response.find(pattern) + len(pattern)
                    end_idx = response.find("\n", start_idx)
                    if end_idx == -1:
                        end_idx = len(response)
                    value = response[start_idx:end_idx].strip()
                    if value:
                        result[field] = value
                        break
        
        return result


class ZeroShotPrompter:
    """Main class for handling zero-shot prompts with different LLM providers."""
    
    def __init__(self, provider: Any):
        """
        Initialize the zero-shot prompter.
        
        Args:
            provider: LLM provider instance to use for generating responses
        """
        self.provider = provider
    
    def run(self, 
            task: ZeroShotTask, 
            input_data: Dict[str, Any], 
            model: Optional[str] = None,
            temperature: float = 0.0) -> Any:
        """
        Run a zero-shot task.
        
        Args:
            task: ZeroShotTask instance
            input_data: Dictionary of input data for the task
            model: Optional model identifier
            temperature: Sampling temperature for generation
            
        Returns:
            Parsed task result
        """
        # Create prompt using the task
        prompt = task.create_prompt(**input_data)
        
        # Get response from the provider
        if hasattr(self.provider, 'get_response'):
            response = self.provider.get_response(
                prompt, 
                model=model,
                temperature=temperature
            )
        else:
            # Fall back to a simple completion call if the provider doesn't have get_response
            response = self.provider.complete(prompt, temperature=temperature)
        
        # Parse the response using the task
        return task.parse_response(response)
    
    def classify(self, 
                 text: str, 
                 labels: List[str], 
                 label_type: str = "categories",
                 model: Optional[str] = None,
                 multi_label: bool = False,
                 prompt_template: Optional[PromptTemplate] = None) -> Union[str, List[str]]:
        """
        Convenience method for zero-shot classification.
        
        Args:
            text: Text to classify
            labels: List of possible classification labels
            label_type: Description of what the labels represent
            model: Optional model identifier
            multi_label: Whether multiple labels can be assigned
            prompt_template: Custom prompt template
            
        Returns:
            Classification result (single label or list)
        """
        task = ZeroShotClassification(
            labels=labels, 
            prompt_template=prompt_template,
            multi_label=multi_label
        )
        return self.run(
            task=task,
            input_data={"text": text, "label_type": label_type},
            model=model,
            temperature=0.1  # Slightly higher than 0 for classification
        )
    
    def answer_question(self,
                        question: str,
                        context: Optional[str] = None,
                        model: Optional[str] = None,
                        prompt_template: Optional[PromptTemplate] = None) -> str:
        """
        Convenience method for zero-shot question answering.
        
        Args:
            question: Question to answer
            context: Optional context information
            model: Optional model identifier
            prompt_template: Custom prompt template
            
        Returns:
            Answer string
        """
        task = ZeroShotQA(prompt_template=prompt_template)
        return self.run(
            task=task,
            input_data={"question": question, "context": context},
            model=model,
            temperature=0.4  # More creative for QA
        )
    
    def extract_info(self,
                    text: str,
                    schema: Dict[str, str],
                    model: Optional[str] = None,
                    output_format: str = "json",
                    prompt_template: Optional[PromptTemplate] = None) -> Dict[str, Any]:
        """
        Convenience method for zero-shot information extraction.
        
        Args:
            text: Text to extract information from
            schema: Dictionary of field names and their descriptions
            model: Optional model identifier
            output_format: Format for output ("json" or "text")
            prompt_template: Custom prompt template
            
        Returns:
            Dictionary of extracted information
        """
        task = ZeroShotExtraction(
            schema=schema, 
            output_format=output_format,
            prompt_template=prompt_template
        )
        return self.run(
            task=task,
            input_data={"text": text},
            model=model,
            temperature=0.0  # Precise extraction
        )


# Example usage
if __name__ == "__main__":
    # This is just a simple example. In real use, you would import this from other files
    class DummyProvider:
        def get_response(self, prompt, model=None, temperature=0):
            print(f"Model: {model}, Temp: {temperature}")
            print(f"Prompt: {prompt}")
            # Mock response - in real use this would call an actual LLM
            if "Classify" in prompt:
                return "technology"
            elif "Question" in prompt:
                return "The answer is 42."
            else:
                return '{"name": "John Doe", "age": 30, "occupation": "Engineer"}'
    
    # Initialize the prompter with a provider
    prompter = ZeroShotPrompter(DummyProvider())
    
    # Example classification
    result = prompter.classify(
        "The new iPhone features a powerful A15 chip and improved camera system.",
        labels=["technology", "sports", "politics", "entertainment"],
        model="gpt-4o-mini"
    )
    print(f"Classification result: {result}")
    
    # Example QA
    answer = prompter.answer_question(
        "What is the meaning of life?",
        context="The meaning of life is often debated by philosophers.",
        model="gpt-4o"
    )
    print(f"QA result: {answer}")
    
    # Example extraction
    info = prompter.extract_info(
        "John Doe is a 30-year-old engineer living in New York.",
        schema={
            "name": "Person's full name",
            "age": "Person's age in years",
            "occupation": "Person's job or profession"
        },
        model="gpt-4o"
    )
    print(f"Extraction result: {info}")
