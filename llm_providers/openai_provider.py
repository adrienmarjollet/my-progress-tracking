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
    

    def get_available_models(self):
        return self.models
    

    def get_response(self, question: str, model=None) -> str:
        if not model:
            model = self.default_model
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": question}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error from OpenAI API: {str(e)}"
    

    def classify_theme(self, question: str, categories: list, model=None) -> str:
        if not model:
            model = self.default_model
            
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
            return "other"

            
    def classify_subtheme(self, question: str, main_theme: str, theme_subcategories: list, model=None) -> str:
        if not model:
            model = self.default_model

        if not theme_subcategories:
            return "other"
            
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
            
        except Exception as _:
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
            return "intermediate"  # Default to intermediate on error

    # def classify_difficulty(self, question, model=None):
    #     """
    #     Classify the difficulty level of a question.
        
    #     Args:
    #         question (str): The question to classify
    #         model (str, optional): The model to use for classification
            
    #     Returns:
    #         str: Difficulty level (easy, medium, hard)
    #     """
    #     if not model:
    #         model = self.default_model
            
    #     prompt = f"""
    #     Analyze the following question and classify its difficulty level as one of: easy, medium, or hard.
    #     Consider the following criteria:
    #     - Easy: Basic, straightforward questions that require simple knowledge or operations
    #     - Medium: Questions requiring some analysis, understanding of concepts, or multi-step processes
    #     - Hard: Complex questions requiring deep understanding, advanced concepts, debugging complex issues, etc.
        
    #     Question: {question}
        
    #     Your response should be only one word: easy, medium, or hard.
    #     """

    #     response = self.client.chat.completions.create(
    #         model=model,
    #         messages=[
    #             # {"role": "system", "content": "You analyze text to determine if it contains an error message."},
    #             {"role": "user", "content": prompt}
    #         ],
    #         max_tokens=20,
    #         temperature=0.1
    #     )
        
    #     result = response.choices[0].message.content.strip().lower()
    #     return result == "error"
            
        # except Exception as e:
        #     # Default to assuming it's a question if we can't determine
        #     return False
        
        # try:
        #     response = self._get_completion(prompt, model)
        #     # Extract just the difficulty level and normalize to lowercase
        #     difficulty = response.lower().strip()
            
        #     # Ensure the response is one of the valid categories
        #     valid_difficulties = ["easy", "medium", "hard"]
        #     if difficulty not in valid_difficulties:
        #         # Find the closest match if response is not exact
        #         for valid in valid_difficulties:
        #             if valid in difficulty:
        #                 return valid
        #         return "medium"  # Default if no valid difficulty is detected
            
        #     return difficulty
        # except Exception as e:
        #     print(f"Error classifying difficulty: {e}")
        #     return "unknown"

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
            # Default to assuming it's a question if we can't determine
            return False