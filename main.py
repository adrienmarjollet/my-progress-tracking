import sqlite3
import os

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from typing import Optional
from datetime import datetime
import plotly.express as px
import plotly.io as pio
import pandas as pd

from llm_providers.openai_provider import OpenAIProvider

from config import OPENAI_API_KEY, DEFAULT_PROVIDER, DEFAULT_MODEL, DICT_CATEGORIES

SUBJECT_CATEGORIES = DICT_CATEGORIES.keys()

THEME_ANALYSIS_MODEL = "gpt-4o-mini"  # Dedicated model for theme analysis

DIFFICULTY_ANALYSIS_MODEL = "gpt-4o-mini"

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)#TODO: this is very permissive for now, see later what can be more suitable.

# Setup templates and static files
templates = Jinja2Templates(directory="templates")

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

DB = 'questions.db'

# Initialize LLM providers
providers = {
    'openai': OpenAIProvider(OPENAI_API_KEY),
}

def init_db():
    with sqlite3.connect(DB) as conn:
        # Create the table if it doesn't exist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                theme TEXT,
                subtheme TEXT,
                provider TEXT,
                model TEXT,
                is_error BOOLEAN,
                difficulty TEXT,
                is_error_msg BOOLEAN,
                helpful INTEGER
            )
        """)
        
        # Check if columns exist, if not, add them
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(questions)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        
        if 'theme' not in column_names:
            conn.execute("ALTER TABLE questions ADD COLUMN theme TEXT")
            
        if 'subtheme' not in column_names:
            conn.execute("ALTER TABLE questions ADD COLUMN subtheme TEXT")
        
        if 'provider' not in column_names:
            conn.execute("ALTER TABLE questions ADD COLUMN provider TEXT")
            
        if 'model' not in column_names:
            conn.execute("ALTER TABLE questions ADD COLUMN model TEXT")
            
        if 'is_error' not in column_names:
            conn.execute("ALTER TABLE questions ADD COLUMN is_error BOOLEAN")
            
        if 'difficulty' not in column_names:
            conn.execute("ALTER TABLE questions ADD COLUMN difficulty TEXT")
            
        if 'is_error_msg' not in column_names:
            conn.execute("ALTER TABLE questions ADD COLUMN is_error_msg BOOLEAN")
            
        if 'helpful' not in column_names:
            conn.execute("ALTER TABLE questions ADD COLUMN helpful INTEGER")
            
        conn.commit()

@app.on_event("startup")
async def startup_event():
    init_db()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Get available models for each provider
    available_models = {}
    for provider_name, provider_instance in providers.items():
        if hasattr(provider_instance, 'get_available_models'):
            available_models[provider_name] = provider_instance.get_available_models()
    
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "answer": None, "providers": providers.keys(), "models": available_models}
    )

@app.post("/")
async def handle_question(
    request: Request,
    question: str = Form(...),
    provider: str = Form(DEFAULT_PROVIDER),
    model: str = Form(DEFAULT_MODEL)
):
    
    timestamp = datetime.now()
    
    llm_provider = providers.get(provider)
    
    if llm_provider:
        answer = llm_provider.get_response(question, model=model)
        # Use the dedicated model for theme classification
        theme = llm_provider.classify_theme(question, SUBJECT_CATEGORIES, model=THEME_ANALYSIS_MODEL)
        sub_theme = llm_provider.classify_subtheme(question, theme, DICT_CATEGORIES[theme], model=THEME_ANALYSIS_MODEL)
        # Determine if the question is an error message
        is_error_msg = llm_provider.is_error_message(question, model=THEME_ANALYSIS_MODEL)
        # Determine difficulty of the question (easy, medium, hard)
        difficulty = llm_provider.judge_difficulty_level(question, model=DIFFICULTY_ANALYSIS_MODEL)
        
    else:
        answer = "Selected provider not available."
        theme = "other"
        sub_theme = "unknown"
        is_error_msg = False
        difficulty = "unknown"
    
    with sqlite3.connect(DB) as conn:
        conn.execute(
            "INSERT INTO questions (question, timestamp, theme, subtheme, provider, model, is_error, difficulty, is_error_msg, helpful) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (question, timestamp, theme, sub_theme, provider, model, is_error_msg, difficulty, is_error_msg, None)
        )
        # Get the last insert id
        cursor = conn.cursor()
        cursor.execute("SELECT last_insert_rowid()")
        question_id = cursor.fetchone()[0]
    
    # Get available models for each provider (for the response template)
    available_models = {}
    for provider_name, provider_instance in providers.items():
        if hasattr(provider_instance, 'get_available_models'):
            available_models[provider_name] = provider_instance.get_available_models()
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request, 
            "answer": answer, 
            "providers": providers.keys(), 
            "models": available_models,
            "question_id": question_id
        }
    )

# Add new endpoint for helpfulness feedback
@app.get("/feedback/{question_id}/{helpful}")
async def submit_feedback(question_id: int, helpful: int):
    with sqlite3.connect(DB) as conn:
        conn.execute("UPDATE questions SET helpful = ? WHERE id = ?", (helpful, question_id))
    return {"success": True}

@app.get("/database", response_class=HTMLResponse)
async def database(request: Request):
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, question, timestamp, theme, subtheme, provider, model, is_error, difficulty, is_error_msg, helpful FROM questions ORDER BY timestamp DESC")
        questions = cursor.fetchall()
    return templates.TemplateResponse(
        "database.html",
        {"request": request, "questions": questions}
    )

@app.get("/delete_question/{id}")  # Changed to match the URL in the template
async def delete_question(id: int):
    with sqlite3.connect(DB) as conn:
        conn.execute("DELETE FROM questions WHERE id = ?", (id,))
    return RedirectResponse(url="/database")

@app.get("/clear_database")  # Changed to match the URL in the template
async def clear_database():
    with sqlite3.connect(DB) as conn:
        conn.execute("DELETE FROM questions")
    return RedirectResponse(url="/database")

@app.get("/visualization", response_class=HTMLResponse)
async def visualization(request: Request):
    # Get data from the database
    with sqlite3.connect(DB) as conn:
        df = pd.read_sql_query("SELECT theme, subtheme, is_error, is_error_msg, difficulty, helpful FROM questions WHERE theme IS NOT NULL", conn)
    
    # Count themes and create histogram
    if not df.empty:
        # Count occurrences of each theme
        theme_counts = df['theme'].value_counts().reset_index()
        theme_counts.columns = ['theme', 'count']
        
        # Create the histogram using Plotly
        fig1 = px.bar(theme_counts, x='theme', y='count', 
                     title='Distribution of Question Themes',
                     labels={'theme': 'Theme', 'count': 'Number of Questions'},
                     color='theme')
        
        # Make layout more attractive
        fig1.update_layout(
            xaxis_title="Theme Categories",
            yaxis_title="Count",
            template="plotly_white"
        )
        
        # Create a subtheme visualization
        if 'subtheme' in df.columns and df['subtheme'].notna().any():
            subtheme_counts = df.groupby(['theme', 'subtheme']).size().reset_index(name='count')
            fig2 = px.bar(subtheme_counts, x='subtheme', y='count', color='theme',
                         title='Distribution of Question Subthemes',
                         labels={'subtheme': 'Subtheme', 'count': 'Number of Questions'},
                         barmode='group')
            
            fig2.update_layout(
                xaxis_title="Subtheme Categories",
                yaxis_title="Count",
                template="plotly_white"
            )
            
            # Add error vs non-error visualization
            if 'is_error_msg' in df.columns:
                error_counts = df['is_error_msg'].value_counts().reset_index()
                error_counts.columns = ['is_error_msg', 'count'] 
                error_counts['is_error_msg'] = error_counts['is_error_msg'].map({1: 'Error Message', 0: 'Regular Question'})
                
                fig3 = px.pie(error_counts, values='count', names='is_error_msg',
                             title='Distribution of Error Messages vs Regular Questions')
                
                # Add difficulty visualization
                if 'difficulty' in df.columns and df['difficulty'].notna().any():
                    difficulty_counts = df['difficulty'].value_counts().reset_index()
                    difficulty_counts.columns = ['difficulty', 'count']
                    
                    # Set appropriate order for difficulty levels
                    if not difficulty_counts.empty:
                        order_dict = {'easy': 0, 'medium': 1, 'hard': 2, 'unknown': 3}
                        difficulty_counts['order'] = difficulty_counts['difficulty'].map(order_dict)
                        difficulty_counts = difficulty_counts.sort_values('order')
                        difficulty_counts = difficulty_counts.drop('order', axis=1)
                    
                    fig4 = px.bar(difficulty_counts, x='difficulty', y='count',
                                title='Distribution of Question Difficulty',
                                labels={'difficulty': 'Difficulty Level', 'count': 'Number of Questions'},
                                color='difficulty',
                                color_discrete_map={
                                    'easy': 'green',
                                    'medium': 'orange',
                                    'hard': 'red',
                                    'unknown': 'gray'
                                })
                    
                    fig4.update_layout(
                        xaxis_title="Difficulty Level",
                        yaxis_title="Count",
                        template="plotly_white"
                    )
                    
                    # Convert figures to HTML
                    plot_html = (pio.to_html(fig1, full_html=False) + "<br><br>" + 
                               pio.to_html(fig2, full_html=False) + "<br><br>" +
                               pio.to_html(fig3, full_html=False) + "<br><br>" +
                               pio.to_html(fig4, full_html=False))
                else:
                    plot_html = (pio.to_html(fig1, full_html=False) + "<br><br>" + 
                               pio.to_html(fig2, full_html=False) + "<br><br>" +
                               pio.to_html(fig3, full_html=False))
            else:
                plot_html = pio.to_html(fig1, full_html=False) + "<br><br>" + pio.to_html(fig2, full_html=False)
        else:
            plot_html = pio.to_html(fig1, full_html=False)
        
        # Add helpfulness visualization if data exists
        if 'helpful' in df.columns and df['helpful'].notna().any():
            helpful_counts = df['helpful'].value_counts().reset_index()
            helpful_counts.columns = ['helpful', 'count']
            helpful_counts['helpful'] = helpful_counts['helpful'].map({1: 'Helpful', 0: 'Not Helpful'})
            
            fig_helpful = px.pie(helpful_counts, values='count', names='helpful',
                               title='User Feedback: Helpful vs Not Helpful Responses',
                               color='helpful',
                               color_discrete_map={
                                   'Helpful': 'green',
                                   'Not Helpful': 'red'
                               })
            
            # Add to plots
            plot_html += "<br><br>" + pio.to_html(fig_helpful, full_html=False)
    else:
        plot_html = "<div class='alert alert-info'>No data available for visualization</div>"
    
    return templates.TemplateResponse(
        "visualization.html",
        {"request": request, "plot_html": plot_html}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)