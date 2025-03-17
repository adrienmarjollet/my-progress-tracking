
class FewShotExample:
    """Represents a single example for few-shot prompting."""
    def __init__(self, input_text, output_text):
        self.input_text = input_text
        self.output_text = output_text
    
    def format(self, input_prefix="Input: ", output_prefix="Output: "):
        """Format the example with customizable prefixes."""
        return f"{input_prefix}{self.input_text}\n{output_prefix}{self.output_text}"
    
    def __str__(self):
        return self.format()


class FewShotPrompt:
    """Manages few-shot examples and constructs prompts."""
    def __init__(self, instruction="", examples=None, input_prefix="Input: ", 
                 output_prefix="Output: ", example_separator="\n\n"):
        self.instruction = instruction
        self.examples = examples or []
        self.input_prefix = input_prefix
        self.output_prefix = output_prefix
        self.example_separator = example_separator
    
    def add_example(self, input_text, output_text=None):
        """Add a new example to the few-shot prompt."""
        if isinstance(input_text, FewShotExample):
            self.examples.append(input_text)
        else:
            self.examples.append(FewShotExample(input_text, output_text))
        return self
    
    def remove_example(self, index):
        """Remove an example at the specified index."""
        if 0 <= index < len(self.examples):
            self.examples.pop(index)
        return self
    
    def clear_examples(self):
        """Remove all examples."""
        self.examples = []
        return self
    
    def set_instruction(self, instruction):
        """Set or update the instruction text."""
        self.instruction = instruction
        return self
    
    def format_examples(self):
        """Format all examples using the current prefixes and separator."""
        formatted_examples = [ex.format(self.input_prefix, self.output_prefix) 
                             for ex in self.examples]
        return self.example_separator.join(formatted_examples)
    
    def build_prompt(self, query=None):
        """Build the final prompt with instruction, examples, and optional query."""
        components = []
        
        if self.instruction:
            components.append(self.instruction)
        
        if self.examples:
            if components:
                components.append("")  # Add space after instruction
            components.append(self.format_examples())
        
        if query is not None:
            if components:
                components.append("")  # Add space before query
            components.append(f"{self.input_prefix}{query}")
            components.append(f"{self.output_prefix}")  # Empty output prefix for model to complete
        
        return "\n".join(components)
    
    def __str__(self):
        return self.build_prompt()


class FewShotManager:
    """Manages multiple few-shot prompt templates."""
    def __init__(self):
        self.prompt_templates = {}
    
    def add_template(self, name, prompt_template):
        """Add or update a prompt template."""
        if not isinstance(prompt_template, FewShotPrompt):
            raise TypeError("Template must be a FewShotPrompt instance")
        self.prompt_templates[name] = prompt_template
        return self
    
    def get_template(self, name):
        """Get a prompt template by name."""
        return self.prompt_templates.get(name)
    
    def list_templates(self):
        """List all available template names."""
        return list(self.prompt_templates.keys())
    
    def remove_template(self, name):
        """Remove a template by name."""
        if name in self.prompt_templates:
            del self.prompt_templates[name]
        return self


# Example usage
def example_usage():
    # Create a few-shot prompt for sentiment classification
    sentiment_prompt = FewShotPrompt(
        instruction="Classify the sentiment of the following text as positive, negative, or neutral.",
        input_prefix="Text: ",
        output_prefix="Sentiment: "
    )
    
    # Add examples
    sentiment_prompt.add_example("I love this product, it's amazing!", "positive")
    sentiment_prompt.add_example("This was a terrible experience, I'm very disappointed.", "negative")
    sentiment_prompt.add_example("The product arrived on time and works as expected.", "neutral")
    
    # Generate a prompt with a new query
    final_prompt = sentiment_prompt.build_prompt(
        "The customer service was helpful but the product quality was poor."
    )
    
    print(final_prompt)
    return final_prompt


if __name__ == "__main__":
    example_usage()
