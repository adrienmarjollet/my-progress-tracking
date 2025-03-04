from flask import Flask
from routes.main import main_bp
from routes.visualization import visualization_bp
from routes.database import database_bp

app = Flask(__name__)

# Register blueprints
app.register_blueprint(main_bp)
app.register_blueprint(visualization_bp)
app.register_blueprint(database_bp)

if __name__ == '__main__':
    app.run(debug=True)
