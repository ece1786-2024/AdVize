import pandas as pd
import openai
from datetime import datetime
import time

# OpenAI API key

client = openai.OpenAI(api_key="")# paste openai api key here


def generate_query_and_feedback(user_profile, ad_content, corr_query):
    """
    Generate feedback and justification for a query and ad content.
    """
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
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def calculate_satisfaction_rate(feedback_df):
    """
    Calculate the satisfaction rate based on simulated feedback.
    """
    total_ads = len(feedback_df)
    interested_ads = feedback_df['Simulated Feedback'].sum()
    return interested_ads / total_ads

def main():
    # Load the user persona
    user_info_path = "user_info_shirley.csv"
    user_data = pd.read_csv(user_info_path)
    if user_data.empty:
        raise ValueError("User information file is empty.")
    
    # Assuming there is only one persona
    user_row = user_data.iloc[0]
    user_profile = {
        "Name": user_row["Name"],
        "Age": user_row["Age"],
        "Gender": user_row["Gender"],
        "Location": user_row["Location"],
        "Occupation": user_row["Occupation"],
        "Interests": user_row["Interests"].split(", "),
        "Search_History": user_row["Search_History"].split(", ")
    }

    # Load the input CSV file
    input_csv_path = "query_matched_Llama.csv"
    column_names = ["Query", "Ad_Content"]

    data = pd.read_csv(input_csv_path, header=None, names=column_names)

    # Prepare to store results
    query_results = []
    start_time = time.time()

    for _, row in data.iterrows():
        query = row['Query']
        ad_content = row['Ad_Content']

        print(f"Processing Query: {query}")

        # Generate feedback
        feedback = generate_query_and_feedback(user_profile, ad_content, query)
        print(f"Feedback: {feedback}")

        # Parse feedback
        lines = feedback.split("\n")
        interested = 1 if "Yes" in lines[0] else 0
        justification = lines[1].replace("- Justification: ", "").strip()

        # Save the result
        query_results.append({
            "Name": user_profile['Name'],
            "Query": query,
            "Ad_Content": ad_content,
            "Simulated Feedback": interested,
            "Simulated Justification": justification
        })
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time *= 10
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    # Create a DataFrame with the results
    feedback_df = pd.DataFrame(query_results)


    # Calculate satisfaction rate
    satisfaction_rate = calculate_satisfaction_rate(feedback_df)
    feedback_df['Satisfaction Rate'] = satisfaction_rate
    print(f"Satisfaction Rate: {satisfaction_rate:.2%}")

    # Save the results to a new CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_path = f"user_feedback_Llama_{timestamp}.csv"
    feedback_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    main()
