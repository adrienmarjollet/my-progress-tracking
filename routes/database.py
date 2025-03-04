from flask import Blueprint, render_template

database_bp = Blueprint('database', __name__)

@database_bp.route('/database')
def database():
    return render_template('database.html')
