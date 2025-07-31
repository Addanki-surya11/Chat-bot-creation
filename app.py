from flask import Flask, request, jsonify, render_template, send_from_directory
from chatbot import SimpleChatbot
import json
import os

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

# Initialize the chatbot
chatbot = SimpleChatbot('intents_final.json')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    response = chatbot.get_response(user_message)
    return jsonify({'response': response})

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    app.run(debug=True)
