import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Default LLM provider
DEFAULT_PROVIDER = "openai"
DICT_DEFAULT_MODEL = {
                        "openai": "gpt-4o-mini", 
                        "ollama":"deepseek-r1:latest"
                    }

# Dedicated models for theme analysis and difficulty analysis
THEME_ANALYSIS_MODEL = DICT_DEFAULT_MODEL["openai"]
DIFFICULTY_ANALYSIS_MODEL = DICT_DEFAULT_MODEL["openai"]

DICT_CATEGORIES = {
                   "programming": ["python", "rust", "c++", "frontend", "algorithms", "backend"],
                   "database": ["sql", "nosql", "relational", "document", "graph"],
                   "machine learning": ["supervised", "unsupervised", "reinforcement", "deep learning", "nlp"],
                   "LLM": ["transformers", "fine-tuning", "prompting", "evaluation", "deployment"],
                   "MLOps": ["experiment tracking", "model registry", "pipeline automation", "monitoring", "serving"],
                   "DevOps": ["ci/cd", "containerization", "infrastructure", "monitoring", "cloud"],
                   "bash": ["scripting", "tools", "automation", "system admin"],
                   "SQL": ["queries", "optimization", "schema design", "stored procedures", "indexes"],
                   "physics": ["mechanics", "electromagnetism", "thermodynamics", "quantum", "relativity"],
                   "chemistry": ["organic", "inorganic", "analytical", "physical", "biochemistry"],
                   "mathematics": ["calculus", "linear algebra", "statistics", "discrete math", "number theory"],
                   "history": ["ancient", "medieval", "modern", "contemporary", "regional"],
                   "culture": ["literature", "art", "music", "philosophy", "sociology"],
                   "history of science": ["scientific revolution", "industrial revolution", "information age", "famous scientists"],
                   "psychology": ["cognitive", "behavioral", "developmental", "clinical", "social"],
                   "medicine": ["anatomy", "physiology", "pathology", "pharmacology", "surgery"],
                   "other": ["not yet classified"]
                   }
                


