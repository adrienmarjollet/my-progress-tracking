import sqlite3
from flask import Blueprint, render_template
import plotly.graph_objects as go
from utils.data_aggregation import get_questions_by_week, get_difficulty_by_week

visualization_bp = Blueprint('visualization', __name__)

DB_PATH = "database/questions.db"  # Adjust path if needed

@visualization_bp.route('/visualization')
def visualization():
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Create difficulty by week chart
        difficulty_by_week = get_difficulty_by_week(conn)
        
        # Convert to sorted lists for plotting
        weeks = []
        difficulties = []
        
        # Sort by year and week
        for week_key in sorted(difficulty_by_week.keys(), 
                              key=lambda x: tuple(map(int, x.split('-')[::-1]))):
            weeks.append(week_key)
            difficulties.append(difficulty_by_week[week_key])
        
        # Create the difficulty graph
        difficulty_fig = go.Figure()
        difficulty_fig.add_trace(
            go.Scatter(
                x=weeks, 
                y=difficulties, 
                mode='lines+markers',
                name='Average Difficulty',
                line=dict(color='#DC143C', width=2),
                marker=dict(size=8)
            )
        )
        
        difficulty_fig.update_layout(
            title="Weekly Average Question Difficulty",
            xaxis_title="Week of Year",
            yaxis_title="Average Difficulty Score",
            height=500,
            template="plotly_white",
            hovermode="x unified"
        )
        
        # Convert the figure to HTML
        difficulty_plot_html = difficulty_fig.to_html(full_html=False, include_plotlyjs=False)
        
        return render_template('visualization.html', plot_html=difficulty_plot_html)
        
    except Exception as e:
        return render_template('error.html', error=str(e))
    finally:
        conn.close()
