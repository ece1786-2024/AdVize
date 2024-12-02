from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import gensim.downloader as api

app = Flask(__name__)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to("cpu")

glove_model = api.load("glove-wiki-gigaword-300")

GLOVE_WEIGHT = 0.7
BERT_WEIGHT = 0.3

commercial_ads_data = None

def load_commercial_ads_data(file_path):
    """
    Load and preprocess the commercial ads dataset.
    """
    commercial_ads_data = pd.read_csv(file_path, sep=',', on_bad_lines='skip', header=None)
    commercial_ads_data.columns = commercial_ads_data.iloc[0]
    commercial_ads_data = commercial_ads_data[1:].reset_index(drop=True)
    commercial_ads_data = commercial_ads_data.dropna()
    commercial_ads_data = commercial_ads_data[commercial_ads_data['label'] == '1']

    commercial_ads_data['search_query_bert_embedding'] = commercial_ads_data['search_query_bert_embedding'].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" ") if isinstance(x, str) else x
    )
    commercial_ads_data['query_glove_embedding'] = commercial_ads_data['query_glove_embedding'].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" ") if isinstance(x, str) else x
    )
    commercial_ads_data['query_combined_embedding'] = commercial_ads_data.apply(
        lambda row: np.concatenate([
            row['query_glove_embedding'] * GLOVE_WEIGHT,
            row['search_query_bert_embedding'] * BERT_WEIGHT
        ]),
        axis=1
    )
    return commercial_ads_data

def get_bert_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to("cpu")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0)
    sentence_embedding = embeddings.mean(dim=0)
    return sentence_embedding.cpu().numpy()

def get_glove_embedding(sentence):
    words = sentence.lower().split()
    word_vectors = [glove_model[word] for word in words if word in glove_model]
    if len(word_vectors) == 0:
        return np.zeros(300)
    sentence_embedding = np.mean(word_vectors, axis=0)
    return sentence_embedding

def search_according_to_query(query, commercial_ads_data):
    query_embedding = get_glove_embedding(query) * GLOVE_WEIGHT
    query_embedding_bert = get_bert_embedding(query) * BERT_WEIGHT
    query_combined_embedding = np.concatenate([query_embedding, query_embedding_bert])

    combined_embeddings = np.vstack(commercial_ads_data["query_combined_embedding"].values)
    cos_sim_combined = np.dot(combined_embeddings, query_combined_embedding) / (
        np.linalg.norm(combined_embeddings, axis=1) * np.linalg.norm(query_combined_embedding)
    )
    most_similar_combined_idx = np.argmax(cos_sim_combined)
    most_similar_combined_description = commercial_ads_data.loc[most_similar_combined_idx, "Ads_Description"]
    return most_similar_combined_description

##############################################################################
# Load ads dataset
##############################################################################
@app.route('/load-data', methods=['POST'])
def load_data():
    global commercial_ads_data
    try:
        data = request.get_json()
        file_path = data.get("file_path")

        if not file_path:
            return jsonify({"error": "File path is required"}), 400

        commercial_ads_data = load_commercial_ads_data(file_path)
        return jsonify({"message": "Dataset loaded successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
##############################################################################
# Flask route ad-pushing algorithm
##############################################################################
@app.route('/run-algo', methods=['POST'])
def run_algo():
    try:
        data = request.get_json()
        query = data["query"]

        result = search_according_to_query(query, commercial_ads_data)
        return jsonify({"matched_ad": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
##############################################################################

# file_path = 'train_250k_query_emb.csv'
# commercial_ads_data = load_commercial_ads_data(file_path)
##############################################################################
# health check 
@app.route('/health-check', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200
##############################################################################

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

