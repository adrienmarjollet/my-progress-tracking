import os
import logging
from openai import OpenAI
from .base import LLMProvider

#TODO: revamp the following with langchain structured outputs

class OpenAIProvider(LLMProvider):
    
    def __init__(self, api_key: str):
        
        self.client = OpenAI(api_key=api_key)
        self.models = {
            'chatgpt-4o-latest': 'chatgpt-4o-latest',
            'o1-mini': 'o1-mini',
            'gpt-4o-mini': 'gpt-4o-mini'
        }

        self.default_model = 'gpt-4o-mini'
        
        # Set up logging
        self.logger = logging.getLogger('openai_provider')
        self.logger.setLevel(logging.INFO)
        
        # Ensure logs directory exists
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create file handler which logs to file (overwrite mode)
        log_file = os.path.join(logs_dir, 'openai_provider.log')
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        self.logger.addHandler(file_handler)
        
        self.logger.info("OpenAIProvider initialized")
    
    def get_available_models(self):
        self.logger.info("Getting available models")
        return self.models
    
    def get_response(self, question: str, model=None) -> str:
        if not model:
            model = self.default_model
        
        self.logger.info(f"Getting response using model: {model}")
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": question}
                ]
            )
            self.logger.info("Response received successfully")
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error getting response: {str(e)}")
            return f"Error from OpenAI API: {str(e)}"
        
    def get_chat_response(self, message, history=None, model="gpt-4o-mini"):
        """Get a response in a multi-turn conversation."""
        self.logger.info(f"Getting chat response using model: {model}")
        
        try:
            # Start with system message
            messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
            
            # Add conversation history if provided
            if history and len(history) > 0:
                messages.extend(history)
            
            # Add the new user message
            messages.append({"role": "user", "content": message})
            
            # Get response from OpenAI
            response = self.client.chat.completions.create(
                model=model,
                messages=messages
            )
            
            self.logger.info("Chat response received successfully")
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error getting chat response: {str(e)}")
            return f"Error: {str(e)}"    
    
    def classify_theme(self, question: str, categories: list, model=None) -> str:
        if not model:
            model = self.default_model
            
        self.logger.info(f"Classifying theme using model: {model}")
            
        prompt = f"""Classify the following question into exactly one of these categories: {', '.join(categories)}.
        Only respond with the category name, nothing else.
        
        Question: {question}"""
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a classifier that categorizes questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.3
            )
            
            theme = response.choices[0].message.content.strip().lower()
            # Ensure the theme is one of the valid categories
            if theme not in categories:
                theme = "other"
            return theme
            
        except Exception as e:
            self.logger.error(f"Error classifying theme: {str(e)}")
            return "other"

            
    def classify_subtheme(self, question: str, main_theme: str, theme_subcategories: list, model=None) -> str:
        if not model:
            model = self.default_model

        if not theme_subcategories:
            return "other"
        
        self.logger.info(f"Classifying subtheme using model: {model}")
            
        prompt = f"""For a question that belongs to the '{main_theme}' category, classify it into exactly one of these subcategories: {', '.join(theme_subcategories)}.
        Only respond with the subcategory name, nothing else.
        
        Question: {question}"""
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a classifier that categorizes questions into subcategories."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.3
            )
            
            subtheme = response.choices[0].message.content.strip().lower()
            # Ensure the subtheme is one of the valid subcategories
            if subtheme not in theme_subcategories:
                subtheme = "other"
            return subtheme
            
        except Exception as e:
            self.logger.error(f"Error classifying subtheme: {str(e)}")
            return "other"

    def judge_difficulty_level(self, question: str, model=None) -> str:
        """
        Judge if the question or error message is beginner, intermediate, or advanced difficulty.
        
        Args:
            question: The question or error message to evaluate
            model: Optional model to use for classification
            
        Returns:
            str: "beginner", "intermediate", or "advanced"
        """
        if not model:
            model = self.default_model
        
        self.logger.info(f"Judging difficulty level using model: {model}")
            
        prompt = f"""Analyze the following question or error message and determine if it's a beginner, intermediate, or advanced level programming question.
        Consider the following factors:
        - Beginner: Basic syntax, simple concepts, common errors, fundamental programming ideas
        - Intermediate: Some experience required, involves multiple concepts, framework-specific issues, moderate complexity
        - Advanced: Complex algorithms, system design, performance optimization, advanced debugging, deep technical knowledge
        
        Only respond with one word: "beginner", "intermediate", or "advanced".
        
        Question/Error: {question}"""
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert at evaluating the difficulty level of programming questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.3
            )
            
            difficulty = response.choices[0].message.content.strip().lower()
            # Ensure the response is one of the valid difficulty levels
            if difficulty not in ["beginner", "intermediate", "advanced"]:
                difficulty = "intermediate"  # Default to intermediate if response is unclear
            return difficulty
            
        except Exception as e:
            self.logger.error(f"Error judging difficulty level: {str(e)}")
            return "intermediate"  # Default to intermediate on error

    def is_error_message(self, prompt: str, model=None) -> bool:
        """
        Determines whether the given prompt contains an error message or is a regular question.
        
        Args:
            prompt: The text to analyze
            model: Optional model to use for classification
            
        Returns:
            bool: True if the prompt appears to be an error message, False otherwise
        """
        if not model:
            model = self.default_model
        
        self.logger.info(f"Determining if text is error message using model: {model}")
            
        analysis_prompt = f"""Analyze the following text and determine if it contains an error message or is a regular question.
        Error messages typically include stack traces, error codes, exception details, or explicit error statements.
        
        Only respond with "error" or "question" - nothing else.
        
        Text: {prompt}"""
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    # {"role": "system", "content": "You analyze text to determine if it contains an error message."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=20,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip().lower()
            return result == "error"
            
        except Exception as e:
            self.logger.error(f"Error determining if text is error message: {str(e)}")
            # Default to assuming it's a question if we can't determine
            return False