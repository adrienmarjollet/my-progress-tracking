import requests
import logging
import os
from .base import LLMProvider

class OllamaProvider(LLMProvider):

    def __init__(self, base_url: str = "http://localhost:11434"):

        self.base_url = base_url
        self.models = {
            "gemma3:1b": "gemma3:1b",
            "deepseek-r1:latest": "deepseek-r1:latest", #TODO this version is 7B for deepseek, find a way to display the more informative and readable name instead.
        }
        self.default_model = "deepseek-r1:latest"

        # Set up logging
        self.logger = logging.getLogger('ollama_provider')
        self.logger.setLevel(logging.INFO)
        
        # Ensure logs directory exists
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create file handler which logs to file (overwrite mode)
        log_file = os.path.join(logs_dir, 'ollama_provider.log')
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Initialized OllamaProvider with base URL: {base_url}")
    
    def get_available_models(self):
        self.logger.debug("Getting available models")
        return self.models
    
    def get_response(self, prompt: str, model = None, stream = False) -> str:

        if not model: 
            model = self.default_model
            
        self.logger.info(f"Getting response using model: {model}, stream={stream}")
        self.logger.info(f"Prompt: {prompt[:50]}...")

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": stream}
            )
            self.logger.info(f"ollama response.json() = {response.json()}")
            response_json = response.json()["response"]
            self.logger.debug("Response received from Ollama API")
            return response_json
        except Exception as e:
            self.logger.error(f"Error in get_response: {str(e)}")
            raise

    def get_chat_response(self, message, history=None, model=None):
        """Get a response in a multi-turn conversation."""
        if not model:
            model = self.default_model
            
        self.logger.info(f"Getting chat response using model: {model}")
        self.logger.debug(f"Message: {message[:50]}...")
        
        try:
            # Format messages for Ollama chat API
            messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
            
            # Add conversation history if provided
            if history and len(history) > 0:
                messages.extend(history)
                self.logger.debug(f"Added {len(history)} messages from history")
            
            # Add the new user message
            messages.append({"role": "user", "content": message})
            
            # Get response from Ollama
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages
                }
            )
            
            content = response.json()['message']['content']
            self.logger.debug("Chat response received from Ollama API")
            return content
        except Exception as e:
            self.logger.error(f"Error in get_chat_response: {str(e)}")
            return f"Error: {str(e)}"    
    

    def classify_theme(self, question: str, categories: list, model=None) -> str:
        if not model:
            model = self.default_model
            
        self.logger.info(f"Classifying theme using model: {model}")
        self.logger.debug(f"Question: {question[:50]}..., Categories: {categories}")
        
        prompt = f"""Classify the following question into exactly one of these categories: {', '.join(categories)}.
        Only respond with the category name, nothing else.
        
        Question: {question}"""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a classifier that categorizes questions."},
                        {"role": "user", "content": prompt}
                    ],
                    "options": {
                        "temperature": 0.3
                    }
                }
            )
            
            theme = response.json()['message']['content'].strip().lower()
            # Ensure the theme is one of the valid categories
            if theme not in categories:
                theme = "other"
                self.logger.warning(f"Theme '{theme}' not in valid categories, defaulting to 'other'")
            else:
                self.logger.debug(f"Classified theme: {theme}")
            return theme
            
        except Exception as e:
            self.logger.error(f"Error in classify_theme: {str(e)}")
            return "other"

            
    def classify_subtheme(self, question: str, main_theme: str, theme_subcategories: list, model=None) -> str:
        if not model:
            model = self.default_model

        self.logger.info(f"Classifying subtheme for theme '{main_theme}' using model: {model}")
        self.logger.debug(f"Question: {question[:50]}..., Subcategories: {theme_subcategories}")
            
        if not theme_subcategories:
            self.logger.debug("No subcategories provided, returning 'other'")
            return "other"
            
        prompt = f"""For a question that belongs to the '{main_theme}' category, classify it into exactly one of these subcategories: {', '.join(theme_subcategories)}.
        Only respond with the subcategory name, nothing else.
        
        Question: {question}"""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a classifier that categorizes questions into subcategories."},
                        {"role": "user", "content": prompt}
                    ],
                    "options": {
                        "temperature": 0.3
                    }
                }
            )
            
            subtheme = response.json()['message']['content'].strip().lower()
            # Ensure the subtheme is one of the valid subcategories
            if subtheme not in theme_subcategories:
                subtheme = "other"
                self.logger.warning(f"Subtheme '{subtheme}' not in valid subcategories, defaulting to 'other'")
            else:
                self.logger.debug(f"Classified subtheme: {subtheme}")
            return subtheme
            
        except Exception as e:
            self.logger.error(f"Error in classify_subtheme: {str(e)}")
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
        self.logger.debug(f"Question: {question[:50]}...")
        
        prompt = f"""Analyze the following question or error message and determine if it's a beginner, intermediate, or advanced level programming question.
        Consider the following factors:
        - Beginner: Basic syntax, simple concepts, common errors, fundamental programming ideas
        - Intermediate: Some experience required, involves multiple concepts, framework-specific issues, moderate complexity
        - Advanced: Complex algorithms, system design, performance optimization, advanced debugging, deep technical knowledge
        
        Only respond with one word: "beginner", "intermediate", or "advanced".
        
        Question/Error: {question}"""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are an expert at evaluating the difficulty level of programming questions."},
                        {"role": "user", "content": prompt}
                    ],
                    "options": {
                        "temperature": 0.3
                    }
                }
            )
            
            difficulty = response.json()['message']['content'].strip().lower()
            # Ensure the response is one of the valid difficulty levels
            if difficulty not in ["beginner", "intermediate", "advanced"]:
                self.logger.warning(f"Invalid difficulty '{difficulty}', defaulting to 'intermediate'")
                difficulty = "intermediate"  # Default to intermediate if response is unclear
            else:
                self.logger.debug(f"Judged difficulty level: {difficulty}")
            return difficulty
            
        except Exception as e:
            self.logger.error(f"Error in judge_difficulty_level: {str(e)}")
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
            
        self.logger.info(f"Checking if prompt is an error message using model: {model}")
        self.logger.debug(f"Prompt: {prompt[:50]}...")
        
        analysis_prompt = f"""Analyze the following text and determine if it contains an error message or is a regular question.
        Error messages typically include stack traces, error codes, exception details, or explicit error statements.
        
        Only respond with "error" or "question" - nothing else.
        
        Text: {prompt}"""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": analysis_prompt}
                    ],
                    "options": {
                        "temperature": 0.1
                    }
                }
            )
            
            result = response.json()['message']['content'].strip().lower()
            is_error = result == "error"
            self.logger.debug(f"Classification result: {'error message' if is_error else 'regular question'}")
            return is_error
            
        except Exception as e:
            self.logger.error(f"Error in is_error_message: {str(e)}")
            # Default to assuming it's a question if we can't determine
            return False

    def embed(self, text, model=None):
        """Get embeddings for the provided text.
        
        Args:
            text: The text to embed
            model: Optional embedding model to use
            
        Returns:
            list: The embedding vector
        """
        if not model:
            model = self.default_model
            
        self.logger.info(f"Getting embedding using model: {model}")
        self.logger.debug(f"Text: {text[:50]}...")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": model,
                    "prompt": text
                }
            )
            
            embedding = response.json()['embedding']
            self.logger.debug(f"Received embedding of dimension {len(embedding)}")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Embedding error: {str(e)}")
            return []