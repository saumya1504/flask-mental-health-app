from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoModel, AutoTokenizer
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins to access the API

# Load the dataset
file_path = "mentalhealth.csv"  # Adjust path if necessary
df = pd.read_csv(file_path)
questions = df["Questions"].values
answers = df["Answers"].values

# Load the Hugging Face model
model_dir = "./"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir)
embedder = pipeline("feature-extraction", model=model, tokenizer=tokenizer)

# Generate question embeddings
question_embeddings = []
for q in questions:
    token_embeddings = embedder(q)[0]
    sentence_embedding = np.mean(token_embeddings, axis=0)
    question_embeddings.append(sentence_embedding)
question_embeddings = np.array(question_embeddings)

def find_best_answer(user_question):
    """Find the best answer based on cosine similarity."""
    user_embedding = embedder(user_question)[0]
    sentence_embedding = np.mean(user_embedding, axis=0)
    similarities = cosine_similarity([sentence_embedding], question_embeddings).flatten()
    best_match_index = np.argmax(similarities)
    return answers[best_match_index], similarities[best_match_index]

@app.route('/get_answer', methods=['POST'])
def get_answer():
    """API endpoint to fetch the best answer."""
    user_question = request.json.get("question")
    best_answer, similarity_score = find_best_answer(user_question)
    return jsonify({"answer": best_answer, "similarity": float(similarity_score)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
