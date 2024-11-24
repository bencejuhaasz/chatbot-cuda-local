from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gpt4all import GPT4All
import os

app = Flask(__name__)
CORS(app)

# Initialize the GPT4All model once
MODEL_PATH = "/home/user/venv/chat/Meta-Llama-3-8B-Instruct.Q4_0.gguf"

try:
    print("Loading GPT4All model...")
    gpt4all_model = GPT4All(MODEL_PATH, device="cuda")
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", str(e))
    gpt4all_model = None  # Handle gracefully if loading fails

# Serve the frontend
@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("frontend", path)

# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    if gpt4all_model is None:
        return jsonify({"error": "Model failed to load."}), 500

    data = request.json
    instruction = data.get("instruction", "").strip()
    file_content = data.get("file_content", "").strip()

    if not instruction and not file_content:
        return jsonify({"error": "Instruction or file content required."}), 400

    # Combine instruction and file content
    prompt = f"{instruction}\n\n{file_content}".strip()

    try:
        # Generate response
        response = gpt4all_model.generate(prompt, max_tokens=512)
        return jsonify({"response": response})
    except Exception as e:
        print("Error during generation:", str(e))
        return jsonify({"error": "Failed to generate response."}), 500

if __name__ == "__main__":
    # Run the app with debug mode on but without auto-reload
    app.run(debug=True, use_reloader=False)
