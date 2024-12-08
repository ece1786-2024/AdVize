# from flask import Flask, request, jsonify
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import openai
# import numpy as np
# from transformers import AutoTokenizer, AutoModel
# import faiss

# app = Flask(__name__)
# client = openai.OpenAI(api_key="")

# # global var
# ads_df = None
# tfidf_vectorizer = None
# ads_tfidf_matrix = None


# def clean_ads_data(file_path):
#     commercial_ads_data = pd.read_csv(file_path, sep='\t', on_bad_lines='skip', header=None)

#     # Remove irrelevant data columns like url, image, user query, etc.
#     # Only keep the ads title and description
#     commercial_ads_data = commercial_ads_data.drop(commercial_ads_data.columns[[0, 1, 2, 5, 6, 7, 8, 9]], axis=1)

#     # Rename columns
#     commercial_ads_data.columns = ['Ads_Title', 'Ads_Description']

#     # Combine 'Ads_Title' and 'Ads_Description' into a single 'Ads_Content' column
#     commercial_ads_data['Ads_Content'] = commercial_ads_data.apply(lambda row: ' | '.join(row.values.astype(str)), axis=1)

#     # Drop the individual 'Ads_Title' and 'Ads_Description' columns
#     commercial_ads_data = commercial_ads_data.drop(columns=['Ads_Title', 'Ads_Description'])

#     return commercial_ads_data

# ##############################################################################
# # health check 
# @app.route('/health-check', methods=['GET'])
# def health_check():
#     return jsonify({"status": "ok"}), 200
# ##############################################################################
# ##############################################################################
# # Load ads dataset 
# # def __init__(self, ads_database) --> this part
# ##############################################################################
# @app.route('/load-data', methods=['POST'])
# def load_data():
#     global ads_df, tfidf_vectorizer, ads_tfidf_matrix
#     try:
#         data = request.get_json()
#         file_path = data.get('file_path')
#         print("file path is: ", file_path)

#         # clean and load dataset
#         print("start cleaning!")
#         ads_data = clean_ads_data(file_path)
#         ads_df = pd.DataFrame(ads_data)
#         tfidf_vectorizer = TfidfVectorizer(stop_words='english')
#         ads_tfidf_matrix = tfidf_vectorizer.fit_transform(ads_df['Ads_Content'])

#         return jsonify({"message": "ads data loaded successfully."}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# ##############################################################################
# # Flask route ad-pushing algorithm
# ##############################################################################
# @app.route('/run-algo', methods=['POST'])
# def run_algo():
#     try:
#         global ads_df, tfidf_vectorizer, ads_tfidf_matrix
#         if ads_df is None or tfidf_vectorizer is None or ads_tfidf_matrix is None:
#             raise Exception("Ads data is not loaded. Use /load-data to load it first.")

#         data = request.get_json()
#         query = data['query']

#         # Use TF-IDF to narrow down candidates
#         query_tfidf = tfidf_vectorizer.transform([query])
#         cosine_similarities = cosine_similarity(query_tfidf, ads_tfidf_matrix).flatten()
#         top_k = 50
#         top_indices = cosine_similarities.argsort()[-top_k:][::-1]
#         narrowed_ads = ads_df.iloc[top_indices].copy()

        

#         # Prepare the prompt for GPT-4
#         prompt = f"The user's search query is: \"{query}\". Based on the given ads, identify the most relevant ad that matches the user's query.\nAds:\n"
#         for idx, row in narrowed_ads.iterrows():
#             prompt += f"Ad {idx + 1}: {row['Ads_Content']}\n"
#         prompt += "\nMust return one ad that is the closest match and return the ad content only without any other phrases."

#         response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[{"role": "system", "content": "You are a helpful assistant."},
#                   {"role": "user", "content": prompt}],
#                 max_tokens=300,
#                 temperature=0.0
#             )
#         chosen_ad = response.choices[0].text.strip()

#         return jsonify({"matched_ad": chosen_ad}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5001)


from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os

app = Flask(__name__)
client = openai.OpenAI(api_key="")

ads_df = None
tfidf_vectorizer = None
ads_tfidf_matrix = None



def clean_ads_data(file_path):

    # detect the delimiter and input file format
    with open(file_path, 'r') as f:
        first_line = f.readline()
        if '\t' in first_line:
            delimiter = '\t'  # TSV
            print("File detected as TSV.")
        elif ',' in first_line:
            delimiter = ','  # CSV
            print("File detected as CSV.")
        else:
            raise ValueError("Unknown file format. The file should be CSV or TSV.")

    commercial_ads_data = pd.read_csv(file_path, sep=delimiter, on_bad_lines='skip')

    # ensure there are Ads_Title and Ads_Description columns 
    required_columns = ['Ads_Title', 'Ads_Description']
    if not all(col in commercial_ads_data.columns for col in required_columns):
        raise ValueError(f"The dataset must contain the columns: {', '.join(required_columns)}.")

    # keep relevant columns
    commercial_ads_data = commercial_ads_data[required_columns]

    # combine title and description 
    commercial_ads_data['Ads_Content'] = commercial_ads_data.apply(lambda row: ' | '.join(row.values.astype(str)), axis=1)

    commercial_ads_data = commercial_ads_data.drop(columns=['Ads_Title', 'Ads_Description'])

    return commercial_ads_data



##############################################################################
# health check
############################################################################## 
@app.route('/health-check', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

##############################################################################
# Load ads dataset 
##############################################################################
@app.route('/load-data', methods=['POST'])
def load_data():

    global ads_df, tfidf_vectorizer, ads_tfidf_matrix
    try:
        data = request.get_json()
        file_path = data.get('file_path')

        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError("The specified file path does not exist.")

        # load dataset
        ads_data = clean_ads_data(file_path)
        ads_df = pd.DataFrame(ads_data)
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        ads_tfidf_matrix = tfidf_vectorizer.fit_transform(ads_df['Ads_Content'])

        return jsonify({"message": "Ads data loaded successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

##############################################################################
# Flask route ad-pushing algorithm
##############################################################################
@app.route('/run-algo', methods=['POST'])
def run_algo():

    try:
        global ads_df, tfidf_vectorizer, ads_tfidf_matrix
        if ads_df is None or tfidf_vectorizer is None or ads_tfidf_matrix is None:
            raise Exception("Ads data is not loaded. Use /load-data to load it first.")

        data = request.get_json()
        query = data.get('query')

        if not query:
            raise ValueError("Query is missing in the request.")

        # TF-IDF to narrow down candidates
        query_tfidf = tfidf_vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_tfidf, ads_tfidf_matrix).flatten()
        top_k = 50
        top_indices = cosine_similarities.argsort()[-top_k:][::-1]
        narrowed_ads = ads_df.iloc[top_indices].copy()

        # GPT-4 to select the best ad
        prompt = f"The user's search query is: \"{query}\". Based on the given ads, identify the most relevant ad that matches the user's query.\nAds:\n"
        for idx, row in narrowed_ads.iterrows():
            prompt += f"Ad {idx + 1}: {row['Ads_Content']}\n"
        prompt += "\nMust return one ad that is the closest match and return the ad content only without any other phrases."

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.0
        )
        chosen_ad = response.choices[0].message.content.strip()

        return jsonify({"matched_ad": chosen_ad}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
