
from few_shots import FewShotExample, FewShotPrompt, FewShotManager

def sentiment_analysis_example():
    """Example of few-shot prompting for sentiment analysis."""
    print("\n=== Sentiment Analysis Few-Shot Example ===\n")
    
    prompt = FewShotPrompt(
        instruction="Analyze the sentiment of the following text as positive, negative, or neutral.",
        input_prefix="Text: ",
        output_prefix="Sentiment: "
    )
    
    # Add examples
    prompt.add_example("I absolutely loved the movie! The acting was superb.", "positive")
    prompt.add_example("The service was terrible and the food was cold.", "negative")
    prompt.add_example("The package arrived on time as expected.", "neutral")
    
    # Query to classify
    query = "While the staff was friendly, the room was dirty and the amenities were outdated."
    
    # Generate and print the complete prompt
    full_prompt = prompt.build_prompt(query)
    print(full_prompt)
    print("\nExpected result: The model would likely classify this as negative.")


def code_completion_example():
    """Example of few-shot prompting for code completion."""
    print("\n=== Code Completion Few-Shot Example ===\n")
    
    prompt = FewShotPrompt(
        instruction="Complete the Python function based on the function name and comments.",
        input_prefix="Function signature: ",
        output_prefix="Implementation: ",
        example_separator="\n\n---\n\n"  # Custom separator between examples
    )
    
    # Add coding examples
    prompt.add_example(
        "def calculate_average(numbers):\n    # Calculate the average of a list of numbers",
        "def calculate_average(numbers):\n    # Calculate the average of a list of numbers\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)"
    )
    
    prompt.add_example(
        "def is_palindrome(text):\n    # Check if the text reads the same backward as forward",
        "def is_palindrome(text):\n    # Check if the text reads the same backward as forward\n    text = text.lower().replace(' ', '')\n    return text == text[::-1]"
    )
    
    # Query for code completion
    query = "def fibonacci(n):\n    # Return the nth number in the Fibonacci sequence"
    
    # Generate and print the complete prompt
    full_prompt = prompt.build_prompt(query)
    print(full_prompt)


def translation_example():
    """Example of few-shot prompting for language translation."""
    print("\n=== Translation Few-Shot Example ===\n")
    
    prompt = FewShotPrompt(
        instruction="Translate the following English text to French.",
        input_prefix="English: ",
        output_prefix="French: "
    )
    
    # Add translation examples
    prompt.add_example("Hello, how are you?", "Bonjour, comment allez-vous?")
    prompt.add_example("I would like to order a coffee, please.", "Je voudrais commander un café, s'il vous plaît.")
    prompt.add_example("Where is the train station?", "Où est la gare?")
    
    # Query to translate
    query = "I'm looking forward to visiting Paris next summer."
    
    # Generate and print the complete prompt
    full_prompt = prompt.build_prompt(query)
    print(full_prompt)


def demonstrate_few_shot_manager():
    """Example of using the FewShotManager to organize multiple prompt templates."""
    print("\n=== FewShotManager Example ===\n")
    
    manager = FewShotManager()
    
    # Create and add templates
    sentiment_template = FewShotPrompt(
        instruction="Classify the sentiment as positive, negative, or neutral.",
        input_prefix="Text: ",
        output_prefix="Sentiment: "
    )
    sentiment_template.add_example("This product exceeded my expectations!", "positive")
    sentiment_template.add_example("I regret making this purchase.", "negative")
    
    translation_template = FewShotPrompt(
        instruction="Translate English to Spanish.",
        input_prefix="English: ",
        output_prefix="Spanish: "
    )
    translation_template.add_example("Good morning", "Buenos días")
    translation_template.add_example("Thank you very much", "Muchas gracias")
    
    # Add templates to manager
    manager.add_template("sentiment", sentiment_template)
    manager.add_template("translation", translation_template)
    
    # List available templates
    print("Available templates:", manager.list_templates())
    
    # Use a template from the manager
    sentiment_prompt = manager.get_template("sentiment")
    if sentiment_prompt:
        print("\nUsing sentiment template:")
        print(sentiment_prompt.build_prompt("The customer service was excellent but the product arrived late."))
    
    translation_prompt = manager.get_template("translation")
    if translation_prompt:
        print("\nUsing translation template:")
        print(translation_prompt.build_prompt("I need help with my luggage."))


def custom_example_formatting():
    """Example of customizing the format of examples."""
    print("\n=== Custom Example Formatting ===\n")
    
    # Create custom examples
    example1 = FewShotExample("What is the capital of France?", "The capital of France is Paris.")
    example2 = FewShotExample("What is the largest planet in our solar system?", "The largest planet in our solar system is Jupiter.")
    
    # Create a prompt with custom formatting
    prompt = FewShotPrompt(
        instruction="Answer the following questions with accurate information.",
        input_prefix="Q: ",
        output_prefix="A: ",
        example_separator="\n-----\n"
    )
    
    # Add the custom examples
    prompt.add_example(example1)
    prompt.add_example(example2)
    
    # Query
    query = "What is the tallest mountain on Earth?"
    
    # Generate and print the complete prompt
    full_prompt = prompt.build_prompt(query)
    print(full_prompt)


if __name__ == "__main__":
    print("Few-Shot Prompting Examples")
    print("==========================")
    
    sentiment_analysis_example()
    code_completion_example()
    translation_example()
    demonstrate_few_shot_manager()
    custom_example_formatting()
    
    print("\n\nAll examples completed!")
