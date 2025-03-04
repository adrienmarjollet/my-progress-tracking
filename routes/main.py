from flask import Blueprint, render_template, request
from utils.formatting import format_code_blocks

main_bp = Blueprint('main', __name__)

# Define providers and models
providers = ['openai']
models = {
    'openai': {
        'chatgpt-4o-latest': 'chatgpt-4o-latest',
        'o1-mini': 'o1-mini',
        'gpt-4o-mini': 'gpt-4o-mini'
    }
}

@main_bp.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    if request.method == 'POST':
        question = request.form.get('question')
        provider = request.form.get('provider')
        model = request.form.get('model')
        
        # Get response from LLM provider
        # This would contain your LLM integration code
        # Placeholder for demonstration
        answer = "This is a sample response."
        
        # Process the answer to format code blocks with HTML
        answer = format_code_blocks(answer)
        
        # Save to database
        # Database saving code would go here
    
    return render_template('index.html', answer=answer, providers=providers, models=models)
