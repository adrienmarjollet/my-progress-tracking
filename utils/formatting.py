import re

def format_code_blocks(text):
    """Format code blocks in markdown style to HTML with proper syntax highlighting classes."""
    if not text:
        return text
        
    # Replace markdown code blocks with HTML code blocks
    # Pattern matches ```language\ncode\n``` or just ```\ncode\n```
    pattern = r'```(\w*)\n([\s\S]*?)\n```'
    
    def replacement(match):
        language = match.group(1) or 'text'
        code = match.group(2)
        return f'<pre><code class="language-{language}">{code}</code></pre>'
    
    formatted_text = re.sub(pattern, replacement, text)
    
    # Also handle inline code with single backticks
    inline_pattern = r'`([^`]+?)`'
    formatted_text = re.sub(inline_pattern, r'<code>\1</code>', formatted_text)
    
    return formatted_text
