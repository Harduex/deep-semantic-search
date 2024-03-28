from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

ollama_llm_model = os.getenv('OLLAMA_LLM_MODEL')

# Example route
@app.route('/', methods=['GET'])
def hello_world():
    return jsonify(message=ollama_llm_model)

# Example route with query parameter
@app.route('/greet', methods=['GET'])
def greet():
    name = request.args.get('name')
    if name:
        return jsonify(message=f'Hello, {name}!')
    else:
        return jsonify(error='Name parameter is missing.')

# Example route with request body
@app.route('/add', methods=['POST'])
def add_numbers():
    data = request.get_json()
    if 'num1' in data and 'num2' in data:
        num1 = data['num1']
        num2 = data['num2']
        result = num1 + num2
        return jsonify(result=result)
    else:
        return jsonify(error='Invalid request body.')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)