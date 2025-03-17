import sqlite3
import os
import logging
import json

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager

from typing import Optional
from datetime import datetime
import plotly.express as px
import plotly.io as pio
import pandas as pd
import markdown  # Add markdown library
from markdown.extensions import codehilite, fenced_code, tables

from llm_providers.openai_provider import OpenAIProvider
from llm_providers.ollama_provider import OllamaProvider

from config import (
    OPENAI_API_KEY,
    DEFAULT_PROVIDER,
    DICT_DEFAULT_MODEL,
    THEME_ANALYSIS_MODEL,
    DIFFICULTY_ANALYSIS_MODEL,
    DICT_CATEGORIES
)

from utils.embedding_models import get_embedding

# Configure basic logging
logging.basicConfig(
    filename = "app.log",
    filemode = "w",
    level=logging.INFO,  # Set the threshold level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

# Create a logger
logger = logging.getLogger(__name__)

SUBJECT_CATEGORIES = DICT_CATEGORIES.keys()

DB = 'questions.db'

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
        
        # Create conversations table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                provider TEXT,
                model TEXT,
                theme TEXT,
                is_user BOOLEAN,
                message TEXT NOT NULL,
                helpful INTEGER
            )
        """)
        
        # Create question_embeddings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS question_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                embeddings TEXT NOT NULL
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

# @app.on_event("startup")
# async def startup_event():
#     init_db()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    init_db()
    logger.info("init_db() initialized from lifespan function with asynxcontextmanager")
    yield

app = FastAPI(lifespan=lifespan)

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


# Initialize LLM providers
providers = {
    'openai': OpenAIProvider(OPENAI_API_KEY),
    'ollama': OllamaProvider(),
}



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

# Add a function to convert markdown to HTML
def convert_markdown_to_html(md_text):
    # Pre-process LaTeX equations to protect them from markdown processing
    # Save inline math: $...$
    import re
    
    # Function to escape special characters in LaTeX
    def escape_latex(match):
        content = match.group(1)
        # Return the match with a special marker to find it later
        return f"INLINE_MATH_START{content}INLINE_MATH_END"
    
    # Function to escape block math equations
    def escape_block_latex(match):
        content = match.group(1)
        # Return the match with a special marker to find it later
        return f"BLOCK_MATH_START{content}BLOCK_MATH_END"
    
    # Temporarily replace LaTeX equations with markers
    md_text = re.sub(r'\$\$(.*?)\$\$', escape_block_latex, md_text, flags=re.DOTALL)
    md_text = re.sub(r'\$(.*?)\$', escape_latex, md_text)
    
    # Configure markdown extensions for better rendering
    extensions = [
        'fenced_code',        # For code blocks with ```
        'codehilite',         # For syntax highlighting
        'tables',             # For table support
        'nl2br',              # Convert newlines to <br>
        'sane_lists',         # Better list handling
    ]
    
    # Convert markdown to HTML
    html = markdown.markdown(md_text, extensions=extensions)
    
    # Restore inline LaTeX equations
    html = re.sub(r'INLINE_MATH_START(.*?)INLINE_MATH_END', r'$\1$', html)
    
    # Restore block LaTeX equations with proper styling
    html = re.sub(r'BLOCK_MATH_START(.*?)BLOCK_MATH_END', r'<div class="math-block">$$\1$$</div>', html)
    
    return html

@app.post("/")
async def handle_question(
    request: Request,
    question: str = Form(...),
    provider: str = Form(DEFAULT_PROVIDER),
    model: str = Form(DICT_DEFAULT_MODEL[DEFAULT_PROVIDER])
):
    
    timestamp = datetime.now()
    
    llm_provider = providers.get(provider)

    logging.info(f"llm_provider: {llm_provider}")
    
    if llm_provider:
        answer = llm_provider.get_response(question, model=model)
        logging.info(f"answer : {answer}")
        # Convert markdown answer to HTML
        answer = convert_markdown_to_html(answer)
        # Use the dedicated model for theme classification
        theme = llm_provider.classify_theme(question, SUBJECT_CATEGORIES, model=THEME_ANALYSIS_MODEL)
        sub_theme = llm_provider.classify_subtheme(question, theme, DICT_CATEGORIES[theme], model=THEME_ANALYSIS_MODEL)
        # Determine if the question is an error message
        is_error_msg = llm_provider.is_error_message(question, model=THEME_ANALYSIS_MODEL)
        # Determine difficulty of the question (easy, medium, hard)
        difficulty = llm_provider.judge_difficulty_level(question, model=DIFFICULTY_ANALYSIS_MODEL)
        # compute embedding of the question and store it in the db.
        question_embedding = get_embedding(question)
        
        # Retrieve similar questions from the database
        similar_questions = []
        with sqlite3.connect(DB) as conn:
            # Get all previous questions with their embeddings
            cursor = conn.cursor()
            cursor.execute("SELECT id, question, embedding, timestamp FROM questions WHERE question != ?", (question,))
            previous_questions = cursor.fetchall()
            
            # Convert string embeddings back to lists of floats
            vector_db = []
            question_timestamps = {}  # Store timestamps for each question
            
            for q_id, q_text, q_embedding, q_timestamp in previous_questions:
                try:
                    # Parse the embedding from string representation to list
                    embedding_list = json.loads(q_embedding.replace("'", "\"")) if q_embedding else None
                    if embedding_list:
                        vector_db.append(({"id": q_id, "text": q_text}, embedding_list))
                        question_timestamps[q_id] = q_timestamp  # Save the timestamp
                except (json.JSONDecodeError, AttributeError) as e:
                    logging.error(f"Error parsing embedding for question {q_id}: {e}")
            
            # Find similar questions if we have any in the database
            if vector_db:
                from utils.embedding_models import retrieve_n_closest_vectors
                similar_results = retrieve_n_closest_vectors(question, vector_db, top_n=2)
                
                for result in similar_results:
                    q_id = result[0]["id"]
                    q_text = result[0]["text"]
                    similarity = round(result[1] * 100, 2)
                    
                    # Calculate days elapsed
                    days_elapsed = None
                    if q_id in question_timestamps:
                        try:
                            # Parse the timestamp
                            question_date = datetime.strptime(question_timestamps[q_id], "%Y-%m-%d %H:%M:%S.%f")
                            days_elapsed = (timestamp - question_date).days
                        except ValueError:
                            # Try without microseconds if that format fails
                            try:
                                question_date = datetime.strptime(question_timestamps[q_id], "%Y-%m-%d %H:%M:%S")
                                days_elapsed = (timestamp - question_date).days
                            except:
                                logging.error(f"Could not parse timestamp for question {q_id}")
                    
                    similar_questions.append({
                        "id": q_id, 
                        "text": q_text, 
                        "similarity": similarity,
                        "days_elapsed": days_elapsed
                    })

    # Store the question with its embedding
    with sqlite3.connect(DB) as conn:
        conn.execute(
            "INSERT INTO questions (question, timestamp, theme, subtheme, provider, model, is_error, difficulty, is_error_msg, helpful, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (question, timestamp, theme, sub_theme, provider, model, is_error_msg, difficulty, is_error_msg, None, json.dumps(question_embedding))
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
            "question_id": question_id,
            "similar_questions": similar_questions  # Pass similar questions to the template
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
        cursor.execute("SELECT id, question, timestamp, theme, subtheme, provider, model, is_error, difficulty, is_error_msg, helpful, embedding FROM questions ORDER BY timestamp DESC")
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
        df = pd.read_sql_query("SELECT theme, subtheme, is_error, is_error_msg, difficulty, helpful, timestamp FROM questions WHERE theme IS NOT NULL", conn)
    
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
        
        # Create the daily question type ratios chart
        if 'difficulty' in df.columns and 'timestamp' in df.columns and df['difficulty'].notna().any():
            # Convert timestamp to date
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            
            # Ensure all difficulty levels have standard names
            difficulty_map = {
                'easy': 'Beginner',
                'medium': 'Intermediate',
                'hard': 'Advanced',
                'unknown': 'Unknown'
            }
            df['difficulty'] = df['difficulty'].map(difficulty_map).fillna('Unknown')
            
            # Group by date and difficulty, then calculate counts
            daily_counts = df.groupby(['date', 'difficulty']).size().unstack().fillna(0)
            
            # If some difficulty levels are missing, add them with zeros
            for level in ['Beginner', 'Intermediate', 'Advanced', 'Unknown']:
                if level not in daily_counts.columns:
                    daily_counts[level] = 0
            
            # Calculate the daily ratios
            daily_total = daily_counts.sum(axis=1)
            daily_ratios = daily_counts.div(daily_total, axis=0) * 100
            
            # Reset index to make date a column
            daily_ratios = daily_ratios.reset_index()
            
            # Create the area chart
            fig_daily_ratio = px.area(
                daily_ratios, 
                x='date', 
                y=['Beginner', 'Intermediate', 'Advanced'],
                title='Daily Question Difficulty Ratios',
                labels={'value': 'Percentage', 'date': 'Date', 'variable': 'Difficulty Level'},
                color_discrete_map={
                    'Beginner': 'green',
                    'Intermediate': 'orange',
                    'Advanced': 'red'
                }
            )
            
            fig_daily_ratio.update_layout(
                xaxis_title="Date",
                yaxis_title="Percentage (%)",
                template="plotly_white",
                hovermode="x unified",
                legend_title="Difficulty Level"
            )
            daily_ratio_plot = pio.to_html(fig_daily_ratio, full_html=False)

            # Build a new line chart for daily difficulty ratios
            melted_ratios = daily_ratios.melt(id_vars=["date"], var_name="Difficulty", value_name="Ratio")
            fig_difficulty_ratio = px.line(
                melted_ratios,
                x="date",
                y="Ratio",
                color="Difficulty",
                title="Daily Ratio of Question Difficulty Over Time (Days)"
            )
            fig_difficulty_ratio.update_layout(
                xaxis_title="Date",
                yaxis_title="Ratio (%)",
                template="plotly_white",
                hovermode="x unified",
                legend_title="Difficulty"
            )
            difficulty_ratio_line_plot = pio.to_html(fig_difficulty_ratio, full_html=False)
        else:
            daily_ratio_plot = "<div class='alert alert-info'>No difficulty data available for daily ratio visualization</div>"
            difficulty_ratio_line_plot = "<div class='alert alert-info'>No difficulty data available for difficulty ratio line chart</div>"
    else:
        plot_html = "<div class='alert alert-info'>No data available for visualization</div>"
        daily_ratio_plot = "<div class='alert alert-info'>No data available for visualization</div>"
        difficulty_ratio_line_plot = "<div class='alert alert-info'>No difficulty data available for difficulty ratio line chart</div>"
    
    return templates.TemplateResponse(
        "visualization.html",
        {
            "request": request, 
            "plot_html": plot_html, 
            "daily_ratio_plot": daily_ratio_plot,
            "difficulty_ratio_line_plot": difficulty_ratio_line_plot
        }
    )

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, conversation_id: str = None):
    # Get available models for each provider
    available_models = {}
    for provider_name, provider_instance in providers.items():
        if hasattr(provider_instance, 'get_available_models'):
            available_models[provider_name] = provider_instance.get_available_models()
    
    messages = []
    if conversation_id:
        # Retrieve existing conversation
        with sqlite3.connect(DB) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT is_user, message, timestamp FROM conversations WHERE conversation_id = ? ORDER BY timestamp", (conversation_id,))
            messages = cursor.fetchall()
    else:
        # Generate a new conversation ID
        import uuid
        conversation_id = str(uuid.uuid4())
    
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request, 
            "conversation_id": conversation_id,
            "messages": messages,
            "providers": providers.keys(), 
            "models": available_models
        }
    )

@app.post("/chat/{conversation_id}")
async def chat_message(
    request: Request,
    conversation_id: str,
    message: str = Form(...),
    provider: str = Form(DEFAULT_PROVIDER),
    model: str = Form(DICT_DEFAULT_MODEL[DEFAULT_PROVIDER])
):
    timestamp = datetime.now()
    llm_provider = providers.get(provider)
    
    # Save user message to the database
    with sqlite3.connect(DB) as conn:
        conn.execute(
            "INSERT INTO conversations (conversation_id, timestamp, provider, model, is_user, message) VALUES (?, ?, ?, ?, ?, ?)",
            (conversation_id, timestamp, provider, model, True, message)
        )
        conn.commit()
    
    # Get LLM response
    if llm_provider:
        # Retrieve conversation history for context
        with sqlite3.connect(DB) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT is_user, message FROM conversations WHERE conversation_id = ? ORDER BY timestamp", (conversation_id,))
            history = cursor.fetchall()
            
        # Format history for the LLM
        formatted_history = []
        for is_user, msg in history:
            role = "user" if is_user else "assistant"
            formatted_history.append({"role": role, "content": msg})
        
        # Get response from LLM with conversation history
        llm_response = llm_provider.get_chat_response(message, formatted_history, model=model)
        
        # Convert markdown response to HTML
        llm_response_html = convert_markdown_to_html(llm_response)
        
        # Classify the theme of the conversation
        if len(formatted_history) <= 1:  # Only for the first interaction
            theme = llm_provider.classify_theme(message, SUBJECT_CATEGORIES, model=THEME_ANALYSIS_MODEL)
        else:
            # Get the existing theme
            with sqlite3.connect(DB) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT theme FROM conversations WHERE conversation_id = ? AND theme IS NOT NULL LIMIT 1", (conversation_id,))
                result = cursor.fetchone()
                theme = result[0] if result else llm_provider.classify_theme(message, SUBJECT_CATEGORIES, model=THEME_ANALYSIS_MODEL)
    else:
        llm_response = "Selected provider not available."
        llm_response_html = llm_response
        theme = "other"
    
    # Save original markdown response to the database
    with sqlite3.connect(DB) as conn:
        conn.execute(
            "INSERT INTO conversations (conversation_id, timestamp, provider, model, theme, is_user, message) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (conversation_id, datetime.now(), provider, model, theme, False, llm_response)
        )
        conn.commit()
    
    # Retrieve the updated conversation
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT is_user, message, timestamp FROM conversations WHERE conversation_id = ? ORDER BY timestamp", (conversation_id,))
        messages = cursor.fetchall()
    
    # Convert all assistant messages to HTML for display
    processed_messages = []
    for is_user, msg, ts in messages:
        if not is_user:
            processed_messages.append((is_user, convert_markdown_to_html(msg), ts))
        else:
            processed_messages.append((is_user, msg, ts))
    
    # Get available models for the response template
    available_models = {}
    for provider_name, provider_instance in providers.items():
        if hasattr(provider_instance, 'get_available_models'):
            available_models[provider_name] = provider_instance.get_available_models()
    
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "conversation_id": conversation_id,
            "messages": processed_messages,
            "providers": providers.keys(),
            "models": available_models
        }
    )

@app.get("/chat_feedback/{conversation_id}/{message_timestamp}/{helpful}")
async def chat_feedback(conversation_id: str, message_timestamp: str, helpful: int):
    with sqlite3.connect(DB) as conn:
        conn.execute("UPDATE conversations SET helpful = ? WHERE conversation_id = ? AND timestamp = ?", 
                    (helpful, conversation_id, message_timestamp))
    return {"success": True}

@app.get("/conversations", response_class=HTMLResponse)
async def view_conversations(request: Request):
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        # Get distinct conversations with their first message and last update time
        cursor.execute("""
            SELECT 
                c1.conversation_id, 
                (SELECT message FROM conversations WHERE conversation_id = c1.conversation_id AND is_user = 1 ORDER BY timestamp LIMIT 1) as first_message,
                MIN(c1.timestamp) as start_time,
                MAX(c1.timestamp) as last_update,
                (SELECT theme FROM conversations WHERE conversation_id = c1.conversation_id AND theme IS NOT NULL LIMIT 1) as theme,
                (SELECT provider FROM conversations WHERE conversation_id = c1.conversation_id LIMIT 1) as provider,
                (SELECT model FROM conversations WHERE conversation_id = c1.conversation_id LIMIT 1) as model,
                COUNT(*) as message_count
            FROM conversations c1
            GROUP BY c1.conversation_id
            ORDER BY last_update DESC
        """)
        conversations = cursor.fetchall()
    
    return templates.TemplateResponse(
        "conversations.html",  # You'll need to create this template
        {"request": request, "conversations": conversations}
    )

@app.get("/delete_conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    with sqlite3.connect(DB) as conn:
        conn.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
    return RedirectResponse(url="/conversations")

@app.get("/clear_conversations")
async def clear_conversations():
    with sqlite3.connect(DB) as conn:
        conn.execute("DELETE FROM conversations")
    return RedirectResponse(url="/conversations")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload = True)