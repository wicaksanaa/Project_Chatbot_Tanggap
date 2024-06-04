from flask import Flask, request, jsonify
import random
from sentence_transformers import SentenceTransformer, util
import json

app = Flask(__name__)


def load_intent_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Let's load the model and intent data
intent_data = load_intent_data('Dataset Chatbot.json')
model = SentenceTransformer('model-bert')

def match_intent(input_token):
    input_embeddings = model.encode(input_token, convert_to_tensor=True)

    best_match = None
    best_similarity = -1  # Perhatikan bahwa similarity akan menjadi nilai -1 hingga 1.

    for intent in intent_data['intents']:
        for pattern in intent['patterns']:
            pattern_embedding = model.encode(pattern, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(input_embeddings, pattern_embedding)[0].item()

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (intent, pattern, similarity)

    return best_match

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        input_user = data['message']
        best_match = match_intent(input_user)
        if best_match is not None:
            matched_intent, matched_pattern, similarity = best_match
            if similarity >= 0.8:
                response = random.choice(matched_intent['response'])
                return jsonify({"response": response}), 200
            else:
                return jsonify({"response": "Mohon maaf chatbot tidak mengerti instruksi dari anda. Mohon berikan instruksi ulang atau berikan instruksi lain."}), 200
        else:
            return jsonify({"response": "Mohon maaf chatbot tidak mengerti instruksi dari anda. Mohon berikan instruksi ulang atau berikan instruksi lain."}), 200
    except Exception as e:
        return jsonify({"response": "Terjadi kesalahan: {}".format(e)}), 500
    
PORT = 5000
if __name__ == "__main__":    # Start the server or listener operation  
    app.run(debug=True, host="0.0.0.0", port=PORT)