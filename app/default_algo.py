'''

    Default algorithm: push_matched_ad()
    Required: a search query (String)
    Output: a matched ad (String)
'''

import sys
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import gensim.downloader as api

# Helper function
def get_bert_embedding(sentence):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to("cpu")
    model = BertModel.from_pretrained("bert-base-uncased").to("cpu")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0)
    sentence_embedding = embeddings.mean(dim=0)
    return sentence_embedding.cpu().numpy()

# Helper function
def get_glove_embedding(sentence):
    words = sentence.lower().split()
    glove_model = api.load("glove-wiki-gigaword-300")
    word_vectors = [glove_model[word] for word in words if word in glove_model]
    if len(word_vectors) == 0:
        return np.zeros(300)
    sentence_embedding = np.mean(word_vectors, axis=0)
    return sentence_embedding

# Find the most relevant ad from the ad database:
def push_matched_ad(query, ad_database, glove_weight=0.7, bert_weight=0.3):

  query_embedding = get_glove_embedding(query) * glove_weight
  query_embedding_bert = get_bert_embedding(query) * bert_weight
  query_combined_embedding = np.concatenate([query_embedding, query_embedding_bert])

  combined_embeddings = np.vstack(ad_database['query_combined_embedding'].values)
  cos_sim_combined = np.dot(combined_embeddings, query_combined_embedding) / (
      np.linalg.norm(combined_embeddings, axis=1) * np.linalg.norm(query_combined_embedding)
  )
  most_similar_combined_idx = np.argmax(cos_sim_combined)
  most_similar_combined_value = cos_sim_combined[most_similar_combined_idx]

  most_similar_combined_description = ad_database.loc[most_similar_combined_idx, 'Ads_Description']
  most_similar_combined_query = ad_database.loc[most_similar_combined_idx, 'Search_Query']

  return most_similar_combined_description

if __name__ == "__main__":

    # Fetch arguments from the command line:
    if len(sys.argv) != 3:
        print("Usage: python default_algo.py <query_string> <ad_db_path>")
        sys.exit(1)
    query_string = sys.argv[1]
    ad_db_path = sys.argv[2]

    # Run the algo to return an ad:
    ads_df = pd.read_csv(ad_db_path, sep=',', on_bad_lines='skip', header = None)
    ad_string = push_matched_ad(
        query_string = query_string,
        ad_database = ads_df
    )
    print(ad_string) # to be captured by `subprocess.check_output()`
