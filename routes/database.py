from flask import Blueprint, render_template, redirect, url_for
import os
import datetime
import sqlite3

database_bp = Blueprint('database', __name__)

def get_db_size_mb(db_path):
    """Get the size of the database file in MB"""
    try:
        size_bytes = os.path.getsize(db_path)
        size_mb = round(size_bytes / (1024 * 1024), 2)
        return size_mb
    except OSError:
        return 0

def get_all_questions():
    """Get all questions from the database"""
    try:
        # Determine the database path - adjust this path as needed
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'questions.db')
        
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Execute query to get all questions
        cursor.execute("SELECT * FROM questions ORDER BY id DESC")
        questions = cursor.fetchall()
        
        # Close connection
        conn.close()
        
        return questions
    except Exception as e:
        print(f"Error retrieving questions: {e}")
        return []

@database_bp.route('/database')
def database():
    # Get all questions from the database
    questions = get_all_questions()
    
    # Debug: Try multiple potential database paths
    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'questions.db'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'questions.db'),
        os.path.abspath('questions.db'),
        'questions.db'
    ]
    
    db_found = False
    for path in possible_paths:
        if os.path.exists(path):
            db_path = path
            db_size = get_db_size_mb(db_path)
            last_update = datetime.datetime.fromtimestamp(os.path.getmtime(db_path)).strftime('%Y-%m-%d %H:%M:%S')
            db_found = True
            print(f"Database found at: {db_path}")
            print(f"Database size: {db_size} MB")
            break
    
    if not db_found:
        db_size = 0
        last_update = "Database file not found"
        print("Database file not found. Checked paths:")
        for path in possible_paths:
            print(f"  - {path}")
    
    # Pass the variables to the template
    return render_template('database.html',
                          questions=questions, 
                          db_size=db_size, 
                          last_update=last_update)

@database_bp.route('/delete_question/<int:question_id>')
def delete_question(question_id):
    try:
        # Determine the database path
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'questions.db')
        
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Delete the question
        cursor.execute("DELETE FROM questions WHERE id = ?", (question_id,))
        conn.commit()
        
        # Close connection
        conn.close()
        
        # Redirect back to database page
        return redirect(url_for('database_bp.database'))
    except Exception as e:
        print(f"Error deleting question {question_id}: {e}")
        # Handle error and redirect
        return redirect(url_for('database_bp.database'))

@database_bp.route('/clear_database')
def clear_database():
    try:
        # Determine the database path
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'questions.db')
        
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Clear all questions
        cursor.execute("DELETE FROM questions")
        conn.commit()
        
        # Close connection
        conn.close()
        
        # Redirect back to database page
        return redirect(url_for('database_bp.database'))
    except Exception as e:
        print(f"Error clearing database: {e}")
        # Handle error and redirect
        return redirect(url_for('database_bp.database'))
