from datetime import datetime
from typing import Dict, List, Tuple, Any
import sqlite3
import json
from .data_validation import validate_metadata

def get_questions_by_week(db_connection: sqlite3.Connection) -> Dict[str, List[Tuple]]:
    """
    Aggregates questions by week number and year from the database.
    
    Args:
        db_connection: SQLite database connection
        
    Returns:
        Dictionary with keys as 'week_number-year' and values as lists of question tuples
    """
    cursor = db_connection.cursor()
    
    # Query to get all questions with timestamps
    cursor.execute("""
        SELECT id, question, timestamp, model_name, metadata 
        FROM questions
        ORDER BY timestamp
    """)
    
    questions_by_week = {}
    
    for question in cursor.fetchall():
        question_id, question_text, timestamp, model_name, metadata = question
        
        # Convert timestamp to datetime
        dt = datetime.fromisoformat(timestamp) if timestamp else datetime.now()
        
        # Get week number and year
        week_number = dt.isocalendar()[1]
        year = dt.year
        
        # Use week-year as key
        key = f"{week_number}-{year}"
        
        if key not in questions_by_week:
            questions_by_week[key] = []
            
        questions_by_week[key].append(question)
    
    return questions_by_week

def get_difficulty_by_week(db_connection: sqlite3.Connection) -> Dict[str, float]:
    """
    Calculates average difficulty score by week from the database.
    
    Args:
        db_connection: SQLite database connection
        
    Returns:
        Dictionary with keys as 'week_number-year' and values as average difficulty scores
    """
    cursor = db_connection.cursor()
    
    # Query to get all questions with timestamps and metadata
    cursor.execute("""
        SELECT timestamp, metadata 
        FROM questions
        WHERE metadata IS NOT NULL
        ORDER BY timestamp
    """)
    
    difficulty_by_week = {}
    count_by_week = {}
    
    for timestamp, metadata in cursor.fetchall():
        # Convert timestamp to datetime
        dt = datetime.fromisoformat(timestamp) if timestamp else datetime.now()
        
        # Get week number and year
        week_number = dt.isocalendar()[1]
        year = dt.year
        
        # Use week-year as key
        key = f"{week_number}-{year}"
        
        # Validate metadata
        metadata_dict = validate_metadata(metadata)
        
        if metadata_dict and 'difficulty' in metadata_dict:
            difficulty = metadata_dict['difficulty']
            
            if key not in difficulty_by_week:
                difficulty_by_week[key] = 0
                count_by_week[key] = 0
            
            difficulty_by_week[key] += difficulty
            count_by_week[key] += 1
    
    # Calculate averages
    avg_difficulty_by_week = {}
    for key in difficulty_by_week:
        if count_by_week[key] > 0:
            avg_difficulty_by_week[key] = round(difficulty_by_week[key] / count_by_week[key], 2)
    
    return avg_difficulty_by_week
