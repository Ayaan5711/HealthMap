import os
import asyncio
from flask import Flask, request, render_template, jsonify, redirect, url_for
from src.ChatBot.chatbot import ingest_data,user_input

app = Flask(__name__)

@app.route('/chatbot')
def chatbot():
    if not os.path.exists("Faiss"):
        ingest_data()
        return redirect(url_for('chatbot'))
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.form['question']
    chat_history = request.form['history']
    response = asyncio.run(user_input(user_question, chat_history))
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)