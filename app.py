from flask import Flask, request, jsonify
from rag import generate_response

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("message")

    if not user_query:
        return jsonify({"reply": "Please enter a valid query."})

    reply = generate_response(user_query)

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
