import sqlite3
import os

def init_db():
    # Ensure the database directory exists
    os.makedirs('database', exist_ok=True)
    
    # Connect to the database (will create it if it doesn't exist)
    conn = sqlite3.connect('database/questions.db')
    cursor = conn.cursor()
    
    # Create questions table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        timestamp TEXT,
        model_name TEXT,
        metadata TEXT
    )
    ''')
    
    # Ensure the metadata column exists, as we need it for difficulty data
    try:
        cursor.execute("SELECT metadata FROM questions LIMIT 1")
    except sqlite3.OperationalError:
        # Add metadata column if it doesn't exist
        cursor.execute("ALTER TABLE questions ADD COLUMN metadata TEXT")
        print("Added metadata column to questions table")
    
    conn.commit()
    conn.close()
    
    print("Database initialization complete")

if __name__ == "__main__":
    init_db()
