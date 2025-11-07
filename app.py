import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Google Gen AI SDK
from google import genai

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="/")
CORS(app)

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("MODEL", "gemini-2.5-flash")

if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in environment/.env")

# Initialize client
client = genai.Client(api_key=API_KEY)  

# A simple system prompt you can tune in AI Studio first
SYSTEM = (
    "You are a concise, friendly support chatbot. "
    "If information is missing, ask a short follow-up."
)

def to_genai_chat(history, user_text):
    """
    Convert our simple history format to Gemini 'contents'.
    history: list of {role: 'user'|'model', content: '...'}
    """
    contents = [
        {"role": "user", "parts": [{"text": f"SYSTEM:\n{SYSTEM}"}]}
    ]
    for m in history:
        contents.append({"role": m["role"], "parts": [{"text": m["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_text}]})
    return contents

@app.route("/")
def serve_ui():
    return send_from_directory(app.static_folder, "index.html")

@app.post("/chat")
def chat():
    """
    JSON body: { "user": "hi", "history": [{role, content}, ...] }
    Returns: { "reply": "..." }
    """
    data = request.get_json(force=True)
    user_text = data.get("user", "").strip()
    history = data.get("history", [])

    if not user_text:
        return jsonify(error="Empty user message"), 400

    try:
        # Text generation with Gemini API
        resp = client.models.generate_content(
            model=MODEL,
            contents=to_genai_chat(history, user_text)
        )  # generateContent API :contentReference[oaicite:3]{index=3}

        reply = resp.text or "(no reply)"
        return jsonify(reply=reply)
    except Exception as e:
        return jsonify(error=str(e)), 500


# (Optional) Streaming endpoint
@app.post("/chat_stream")
def chat_stream():
    """
    Same request shape as /chat, but streams partial tokens.
    Note: In production, consider Server-Sent Events (SSE) or WebSockets.
    """
    from flask import Response

    data = request.get_json(force=True)
    user_text = data.get("user", "").strip()
    history = data.get("history", [])

    def generate():
        stream = client.models.generate_content_stream(
            model=MODEL,
            contents=to_genai_chat(history, user_text)
        )  # streamGenerateContent :contentReference[oaicite:4]{index=4}

        for event in stream:
            if hasattr(event, "text") and event.text:
                yield event.text

    return Response(generate(), mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
