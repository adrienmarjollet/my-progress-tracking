import sys
import os

# Add parent directory to path to import from project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompting.zero_shot import ZeroShotPrompter, PromptTemplate
from llm_providers.openai_provider import OpenAIProvider
from config import OPENAI_API_KEY

def main():
    # Initialize the provider
    provider = OpenAIProvider(OPENAI_API_KEY)
    
    # Initialize the zero-shot prompter
    prompter = ZeroShotPrompter(provider)
    
    print("=== Zero-Shot Prompting Examples ===\n")
    
    # Example 1: Text classification
    print("1. Text Classification Example:")
    text = input("Enter text to classify: ") or "Python is a high-level, interpreted programming language known for its readability."
    categories = ["technology", "health", "entertainment", "sports", "politics", "education"]
    
    result = prompter.classify(
        text=text,
        labels=categories,
        label_type="topics",
        model="gpt-4o-mini"
    )
    
    print(f"Classification result: {result}\n")
    
    # Example 2: Question answering
    print("2. Question Answering Example:")
    question = input("Enter a question: ") or "What are the main benefits of object-oriented programming?"
    
    answer = prompter.answer_question(
        question=question,
        model="gpt-4o-mini"
    )
    
    print(f"Answer: {answer}\n")
    
    # Example 3: Information extraction
    print("3. Information Extraction Example:")
    extract_text = input("Enter text to extract information from: ") or "John Smith is a 35-year-old software engineer who works at Google in Mountain View, California."
    
    schema = {
        "name": "The person's full name",
        "age": "The person's age",
        "occupation": "The person's job or profession",
        "employer": "The company or organization the person works for",
        "location": "The city and/or state where the person works"
    }
    
    info = prompter.extract_info(
        text=extract_text,
        schema=schema,
        model="gpt-4o-mini"
    )
    
    print("Extracted information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Example 4: Using a custom template
    print("\n4. Custom Template Example:")
    custom_template_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "prompting", "templates", "classification_template.txt"
    )
    
    # Create a template from file if it exists, otherwise use a simple string template
    if os.path.exists(custom_template_path):
        custom_template = PromptTemplate.from_file(custom_template_path)
    else:
        custom_template = PromptTemplate(
            "Classify this text: {text}\n"
            "Choose from these labels: {labels}\n"
            "Classification:"
        )
    
    custom_result = prompter.classify(
        text="The stock market plunged yesterday due to concerns about inflation.",
        labels=["finance", "politics", "technology", "sports"],
        prompt_template=custom_template,
        model="gpt-4o-mini"
    )
    
    print(f"Custom template classification: {custom_result}")

if __name__ == "__main__":
    main()
