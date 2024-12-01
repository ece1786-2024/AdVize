from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import gensim.downloader as api
import os
import subprocess
from openai import OpenAI

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

def get_ad_from_docker_image(docker_image_path, query_string, ad_db_file_path):

    # Load Docker image from the .tar file:
    output = subprocess.check_output(["docker", "load", "-i", "algo-image.tar"], text=True).strip()
    image_name = None
    if "Loaded image:" in output:
        image_name = output.split("Loaded image:")[1].strip()

    # Run the Docker image with parameters:
    command = [
        "docker", "run", "--rm",
        image_name, query_string, ad_db_file_path
    ]
    ad_string = subprocess.check_output(command, text=True).strip()
    return ad_string

def generate_query_gpt4(user_profile):
    example_searches = [
        "Portable Wrench Holder", "what do skin tags look like on the face", "fuel gauges",
        "horseshoe purse", "DELL LAPTOP DOCKING STATION TRIPLE MONITOR", "engagement rings",
        "carpet liquidators", "operation gridlock", "moen shower faucet repair", "3t in sneakers",
        "concords 11", "lifeproof", "xbox one +kinect +adapter", "roaring 20s women's swimwear",
    ]

    # Generate a prompt for query generation
    prompt = f"""
        You are tasked with generating 10 realistic and contextually appropriate product search queries for a user profile.

        **User Profile**:
        - Name: {user_profile['Name']}
        - Age: {user_profile['Age']}
        - Gender: {user_profile['Gender']}
        - Location: {user_profile['Location']}
        - Occupation: {user_profile['Occupation']}
        - Interests: {", ".join(user_profile['Interests'])}
        - Recent Searches: {", ".join(user_profile['Search_History'])}

        **Instructions**:
        - Generate 10 realistic and contextually appropriate product search queries for the user profile.
            - Reflect on the user's Interests, Occupation, and Recent Searches.
            - Generate general product search queries relevant to the user’s profile.
            - Generated query should be similar to this example: {", ".join(example_searches)}.
        - Do not include location-specific terms or time-sensitive events in the queries.

        **Output**:
        - A numbered list of 10 queries. Don't output any other content.
    """
        
    client = OpenAI(api_key="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")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def generate_feedback_gpt4(user_profile, input_query, output_ad):
    # Generate a prompt for ad evaluation on the given ad:
    prompt = f"""
        You are tasked with evaluating the relevance of an advertisement for a specific query and user profile.

        **User Profile**:
        - Name: {user_profile['Name']}
        - Age: {user_profile['Age']}
        - Gender: {user_profile['Gender']}
        - Location: {user_profile['Location']}
        - Occupation: {user_profile['Occupation']}
        - Interests: {", ".join(user_profile['Interests'])}
        - Recent Searches: {", ".join(user_profile['Search_History'])}

        **Query**: {input_query}

        **Ad Content**:
        {output_ad}

        **Instructions**:
        - Respond with:
            - Interested: [Yes/No]
            - Justification: [A brief explanation of why this ad would or would not interest the user]
        - Please notice:
            - You may use both Recent Searches and Interests to predict whether this person will like the ad.
            - Generate the response in first person (e.g. 'I am interested in this ...').
            - You may infer the race of the person from the name and also use it as a factor to predict whether this person will like the ad.

        **Output**: (start with '- Interested' and '- Justification')
        - Interested
        - Justification
    """

    client = OpenAI(api_key="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")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def compute_satisfaction_rate(algo_image_path, ad_dataset_path):

    # Load user persona data from CSV file:
    file_path = 'user_info.csv'
    user_data = pd.read_csv(file_path)

    query_results = []
    for index, row in user_data.iterrows():
        # print(index)
        user_profile = {
            "Name": row['Name'],
            "Age": row['Age'],
            "Occupation": row['Occupation'],
            "Location": row['Location'],
            "Gender": row['Gender'],
            "Interests": row['Interests'].split(", "),
            "Search_History": row['Search_History'].split(", ")
        }
        print(f"\nSimulating user: {user_profile['Name']}")

        # Step 1: Generate queries
        print("Generating queries and awaiting ad content...\n")
        queries_and_feedback = generate_query_gpt4(user_profile)
        print(queries_and_feedback)

        queries_list = queries_and_feedback.split("\n")
        commercial_ads_data = load_commercial_ads_data(ad_dataset_path)
        for query in queries_list:
            print('\n')
            ad_content = get_ad_from_docker_image(algo_image_path, query, commercial_ads_data)

            # Step 2: Evaluate ad feedback
            print("__________" + ad_content)
            feedback = generate_feedback_gpt4(
                user_profile = user_profile,
                input_query = query,
                output_ad = ad_content
            )
            print(f"Simulated User Feedback:\n{feedback}")

            #Process feedback
            lines = feedback.split("\n")
            interest_line = [line for line in lines if line.startswith("- Interested")][0]
            justification_line = [line for line in lines if line.startswith("- Justification")][0]
            interest = 1 if "Yes" in interest_line else 0 # converts Yes to 1 and No to 0
            justification = justification_line.replace("- Justification: ", "").strip()

            # Save feedback
            query_results.append({
                "Name": user_profile['Name'],
                "Query": query,
                "Ad_Content": ad_content,
                "Simulated Feedback": interest,
                "Simulated Justification": justification
            })

    feedback_df = pd.DataFrame(query_results)

    # (Optional) Save results to a CSV file:
    # feedback_df.to_csv("user_query_and_feedback.csv", index=False)
    # print("\nResults saved to <user_query_and_feedback.csv>.")

    # Compute satisfaction rate:
    satisfied_count = feedback_df['Simulated Feedback'].sum()
    total_count = feedback_df['Simulated Feedback'].count()
    satisfaction_rate = satisfied_count / total_count
    print(f"Satisfaction Rate: {satisfaction_rate:.2%}")
    return satisfaction_rate



'''
    Flask Backend API Endpoints:
'''
app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        docker_file = request.files.get('docker_image')
        dataset_file = request.files.get('dataset_file')

        if docker_file:
            docker_file_path = os.path.join('algorithms', docker_file.filename)
            docker_file.save(docker_file_path)
        else:
            return jsonify({"error": "Docker image file is missing"}), 400

        if dataset_file:
            dataset_file_path = os.path.join('ad_datasets', dataset_file.filename)
            dataset_file.save(dataset_file_path)
        else:
            return jsonify({"error": "Dataset file is missing"}), 400

        return jsonify({
            "message": "Files uploaded successfully",
            "docker_file_path": docker_file_path,
            "dataset_file_path": dataset_file_path
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['POST'])
def test():
    print("test()")
    return jsonify({
        "message": "Success"
    }), 200

@app.route('/upload-and-run', methods=['POST'])
def run_algorithm():
    print("run_algorithm()")
    try:
        # Save the files on server:
        docker_file = request.files.get('docker_image')
        # dataset_file = request.files.get('dataset_file')

        docker_file_path = os.path.join('algorithms', docker_file.filename)
        docker_file.save(docker_file_path)
        print("Docker image saved at", docker_file_path)
            
        # dataset_file_path = os.path.join('ad_datasets', dataset_file.filename)
        # dataset_file.save(dataset_file_path)
        dataset_file_path = "train_250k_query_emb.csv"

        # Start the Docker container:
        satisfaction_rate = compute_satisfaction_rate(
            algo_image_path = docker_file_path,
            ad_dataset_path = dataset_file_path
        )

        # Simulate algorithm logic (replace with your own)
        algo_result = {
            "satisfaction_rate": satisfaction_rate,
            "message": "Algorithm ran successfully",
            "result": "Simulated result here"
        }
        return jsonify(algo_result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)

