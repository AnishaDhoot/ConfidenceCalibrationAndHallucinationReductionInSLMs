"""
Flask app for the VIT Q&A demo site.

Run from the project root:
    python chatbot_site/app.py
Then open http://127.0.0.1:5000

Use the fine-tuned model instead of the dataset lookup:
    set CHATBOT_BACKEND=lora
    set ADAPTER_PATH=phi3-vit-lora-expc-idk15
    python chatbot_site/app.py
"""

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chatbot import get_bot  # noqa: E402

app = Flask(__name__)
bot = get_bot()

MAX_QUESTION_CHARS = 500


@app.context_processor
def inject_backend():
    return {"backend_label": bot.label}


@app.route("/")
def home():
    return render_template("index.html", page="home")


@app.route("/academics")
def academics():
    return render_template("academics.html", page="academics")


@app.route("/campus-life")
def campus_life():
    return render_template("campus.html", page="campus")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Please type a question."}), 400
    if len(question) > MAX_QUESTION_CHARS:
        return jsonify({"error": f"Please keep questions under {MAX_QUESTION_CHARS} characters."}), 400
    return jsonify(bot.answer(question))


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "backend": bot.name})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
