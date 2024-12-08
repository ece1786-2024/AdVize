import pandas as pd

file_path = "user_feedback_interactive_1.csv"  
data = pd.read_csv(file_path)

data["Persona Owner's feedback to query"] = ""
data["if 1 is given, why"] = ""
data["Persona Owner's feedback"] = ""
data["Reason"] = ""

def process_feedback(feedback):
    if "- Interested: Yes" in feedback:
        return 1, feedback.split("- Justification:")[1].strip()
    elif "- Interested: No" in feedback:
        return 0, feedback.split("- Justification:")[1].strip()
    else:
        return "", ""

data["Simulated Feedback"], data["Simulated Justification"] = zip(*data["Feedback"].apply(process_feedback))

columns = [
    "Name", "Query", "Persona Owner's feedback to query", "if 1 is given, why",
    "Ad_Content", "Simulated Feedback", "Simulated Justification",
    "Persona Owner's feedback", "Reason"
]
transformed_data = data[columns]


output_file = "transformed_file.csv"  
transformed_data.to_csv(output_file, index=False)

print(f"Transformed CSV saved to {output_file}")
