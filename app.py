from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()

from api.home_screen import homescreen_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(homescreen_bp, url_prefix="/api/home_screen")


@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)