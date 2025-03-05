from flask import Flask, render_template
from routes.main import main_bp
from routes.visualization import visualization_bp
from routes.database import database_bp
import os
import datetime

app = Flask(__name__)

# Register blueprints
app.register_blueprint(main_bp)
app.register_blueprint(visualization_bp)
app.register_blueprint(database_bp)

def get_db_size_mb(db_path):
    """Get the size of the database file in MB"""
    try:
        size_bytes = os.path.getsize(db_path)
        size_mb = round(size_bytes / (1024 * 1024), 2)
        return size_mb
    except OSError:
        return 0

if __name__ == '__main__':
    app.run(debug=True)
