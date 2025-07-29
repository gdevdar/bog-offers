from flask import Flask, request, jsonify, send_from_directory
from rag_scripts.offer_recommenderv2 import chat_with_model
import os

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '')
    history = data.get('history', [])
    response = chat_with_model(message, history)
    return jsonify({'response': response})

@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'index.html')   

# Optionally serve other static files (CSS, JS) if needed
@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(os.getcwd(), path)

if __name__ == '__main__':
    app.run(debug=True) 