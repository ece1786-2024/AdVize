def compute_satisfaction_rate(docker_image):
        
    # print("Current working directory:", os.getcwd())

    # user persona data
    user_data_path = "user_info.csv" # load persona csv here
    user_data = pd.read_csv(user_data_path)

    # docker image - ad pushing algo
    docker_url, container_id = start_user_docker(image_name=docker_image)
    print('The docker URL is: ', docker_url)

    # commercial ads data
    ads_data_path = "train_250k_query_emb.csv"
    load_data_to_docker(docker_url, ads_data_path)

    print("Starting user simulation...\n")
    query_results = []
    try:
        for index, row in user_data.iterrows():
            user_profile = {
                "Name": row["Name"],
                "Age": row["Age"],
                "Occupation": row["Occupation"],
                "Location": row["Location"],
                "Gender": row["Gender"],
                "Interests": row["Interests"].split(", "),
                "Search_History": row["Search_History"].split(", ")
            }
            print(f"\nSimulating user: {user_profile['Name']}")

            # generate queries
            queries_list = generate_query_and_feedback(user_profile).split("\n")

            for query in queries_list:
                # print(f"Processing query: {query}")

                # ad content from docker
                ad_content = get_ad_content_from_docker(docker_url, query)
                # print(f"Ad content: {ad_content}")

                # evaluate ad feedback
                feedback = generate_query_and_feedback(user_profile, ad_content=ad_content, corr_query=query)
                # print(f"Feedback: {feedback}")

                lines = feedback.split("\n")
                interest = 1 if "Yes" in lines[0] else 0
                justification = lines[1].replace("- Justification: ", "").strip()

                # Save results
                query_results.append({
                    "Name": user_profile['Name'],
                    "Query": query,
                    "Ad_Content": ad_content,
                    "Simulated Feedback": interest,
                    "Simulated Justification": justification
                })
    finally:
        stop_user_docker(container_id)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"user_feedback_{docker_image}_{timestamp}.csv"
    feedback_df = pd.DataFrame(query_results)
    feedback_df.to_csv(output_path, index=False)
    print(f"Feedback saved to {output_path}")

    # satisfaction rate
    satisfaction_rate = calculate_satisfaction_rate(output_path)
    feedback_df['Satisfaction Rate'] = satisfaction_rate
    feedback_df.to_csv(output_path, index=False)

    print(f"Feedback saved to {output_path}")
    print(f"Satisfaction Rate: {satisfaction_rate:.2%}")

    return satisfaction_rate

import requests
import subprocess
import pandas as pd
import numpy as np
import openai
from openai import OpenAI
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt

client = OpenAI(
    api_key="API_KEY"
) # paste openai api key here

##############################################################################
# Docker Management
##############################################################################

def get_ad_content_from_docker(docker_url, query):
# def get_ad_content_from_docker(query):

    # url = "http://localhost:5001/run-algo"
    url = f"{docker_url}/run-algo"
    print("url in the get ad content: ", url)
    payload = {
        "query": query
    }
    try:
        # response = requests.post(f"{docker_url}/run-algo", json=payload)
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("matched_ad", "No match found")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error from Docker container: {str(e)}")

def load_data_to_docker(docker_url, dataset_path):
    url = f"{docker_url}/load-data"
    payload = {"file_path": dataset_path}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Dataset loaded successfully into Docker container.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to load data into Docker container: {str(e)}")

def start_user_docker(image_name, port=5001):
    try:
        print("Starting the Docker container...\n")
        container_id = subprocess.check_output([
            "docker", "run", "-d", "-p", f"{port}:5001", image_name
        ]).strip().decode("utf-8")
        # current_dir = os.getcwd()  # current working directory
        
        # print("current dir is: ", os.getcwd())
        # container_id = subprocess.check_output([
        #     "docker", "run", "-d", "-p", f"{port}:5001", "-v", f"{current_dir}:/app", image_name
        # ]).strip().decode("utf-8")

        print(f"Started Docker container with ID: {container_id}")
        # check - if flask is ready
        health_check_url = f"http://localhost:{port}/health-check"
        for _ in range(20):  
            try:
                response = requests.get(health_check_url, timeout=2)
                if response.status_code == 200:
                    print("Flask ready!")
                    return f"http://localhost:{port}", container_id
            except requests.exceptions.RequestException:
                print("Waiting for Flask to be ready...")
                time.sleep(10)
        
        raise Exception("Flask app failed to start within the expected time.")
    except Exception as e:
        raise Exception(f"Failed to start Docker container: {e}")

def stop_user_docker(container_id):
    try:
        print(f"Stopping Docker container with ID: {container_id}")
        subprocess.check_call(["docker", "stop", container_id])
        print("Stopped Docker container.")
    except Exception as e:
        print(f"Error stopping Docker container: {e}")

##############################################################################
# Query and Feedback
##############################################################################

def generate_query_and_feedback(user_profile, ad_content=None, corr_query=None, examples=None):
    if examples is None:
        examples = [
            "Portable Wrench Holder", "what do skin tags look like on the face", "fuel gauges",
            "horseshoe purse", "DELL LAPTOP DOCKING STATION TRIPLE MONITOR", "engagement rings",
            "carpet liquidators", "operation gridlock", "moen shower faucet repair", "3t in sneakers",
            "concords 11", "lifeproof", "xbox one +kinect +adapter", "roaring 20s women's swimwear",
        ]

    if ad_content:
        # Generate a prompt for ad evaluation
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

            **Query**: {corr_query}

            **Ad Content**:
            {ad_content}

            **Instructions**:
            - Respond with:
              - Interested: [Yes/No]
              - Justification: [A brief explanation of why this ad would or would not interest the user]
            - Please notice:
              - You need to evaluate the advertisment on both the relevance to the query, but also the interest of the person, if the advertisment is not relevant to the query but is in the interest of the person, it is also considered as interested
              - Also, you can inference the race of the person from the name and also consider it as a factor in the evaluation of the advertisment
              - Generate the response in the first point of view, for example, use 'I will be interested in this' instead of {user_profile['Name']} is interested in this
              - When you making the dicision of interested or not, think in the big picture, think twice on the interests of your persona owner, which are {user_profile['Interests']}, think if the advertisment is aligning with the interest.
              - When you making the dicision of interested or not, think generally about the pushed ads don’t get too hung up on the specific items in a particular ad, focus more on the category and features of the ad. For example, if a user searches for RTX4090, if the pushed advertisement is for an older graphics card such as GT260, the user you are playing may also be potentially interested.
              - When making the decision of whether an ad is interesting or not, please consider the persona's age and the typical preferences of their age group. Think about how someone in this age group might perceive the ad—would it appeal to their lifestyle, interests, or needs?
            **Output**:
            - Interested: [Yes/No]
            - Justification: [Explanation]
        """
    else:
        # Generate a prompt for query generation
        prompt = f"""
            You are tasked with generating 20 realistic and contextually appropriate product search queries for a user profile.

            **User Profile**:
            - Name: {user_profile['Name']}
            - Age: {user_profile['Age']}
            - Gender: {user_profile['Gender']}
            - Location: {user_profile['Location']}
            - Occupation: {user_profile['Occupation']}
            - Interests: {", ".join(user_profile['Interests'])}
            - Recent Searches: {", ".join(user_profile['Search_History'])}

            **Instructions**:
            - Generate 20 realistic and contextually appropriate product search queries for the user profile.
              - Reflect on the user's interests, occupation, and recent searches.
              - Generate general product search queries relevant to the user’s profile.
              - Generated query should be similar to this example: {", ".join(examples)}.
            - Do not include location-specific terms or time-sensitive events in the queries.
            - Make sure the generated query is an object that is reasonable and realistic.

            **Output**:
            - A numbered list of 20 queries. You can only generate this 20 queries, don't output extra words
        """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


##############################################################################
# Satisfaction and plot
##############################################################################

def calculate_satisfaction_rate(csv_path):
    feedback_df = pd.read_csv(csv_path)
    total_ads = len(feedback_df)
    interested_ads = feedback_df['Simulated Feedback'].sum()
    satisfaction_rate = interested_ads / total_ads
    return satisfaction_rate

def plot_satisfaction_rates():
    files = [f for f in os.listdir() if f.startswith('user_feedback_') and f.endswith('.csv')]
    rates = []
    labels = []

    for file in files:
        feedback_df = pd.read_csv(file)
        satisfaction_rate = calculate_satisfaction_rate(file)
        rates.append(satisfaction_rate)
        labels.append(file.replace("user_feedback_", "").replace(".csv", ""))

    plt.figure(figsize=(10, 6))
    plt.bar(labels, rates, alpha=0.7)
    plt.title("Satisfaction Rates Across Ad-Pushing Algorithms")
    plt.ylabel("Satisfaction Rate")
    plt.xlabel("Algorithm + Timestamp")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

