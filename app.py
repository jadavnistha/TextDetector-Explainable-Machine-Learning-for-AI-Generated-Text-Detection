"""
app.py — Flask web app for AI-generated text detection.

Run:  python app.py
Open:  http://127.0.0.1:5001
"""

from flask import Flask, render_template, request, jsonify
from src.predict import predict_document, predict_sentences

app = Flask(__name__)


@app.route("/")
def index():
    """Landing page with hero, features, and CTA."""
    return render_template("index.html", active_page="home")


@app.route("/analyse", methods=["GET"])
def analyse_page():
    """Serve the analysis UI."""
    return render_template("analyse.html", active_page="analyse")


@app.route("/analyse", methods=["POST"])
def analyse_api():
    """Accept JSON { text: "..." }, return document + sentence predictions."""
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    doc_result = predict_document(text)
    sent_results = predict_sentences(text)

    return jsonify({
        "document": doc_result,
        "sentences": sent_results,
    })


@app.route("/about")
def about():
    """About page — methodology, models, tech stack."""
    return render_template("about.html", active_page="about")


@app.route("/contact")
def contact():
    """Contact page — form and FAQ."""
    return render_template("contact.html", active_page="contact")


if __name__ == "__main__":
    print("Models loaded. Starting Flask server...")
    app.run(debug=False, port=5001)
