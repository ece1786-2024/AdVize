from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

app = Flask(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "./llama8B_lora_finetuned"

# Ensure GPU is used if available
device = "mps" if torch.mps.is_available() else "cpu"

# Initialize tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto" if device == "mps" else None,
        load_in_8bit=device == "mps",
        local_files_only=True 
    )
    model = model.to(device)
except Exception as e:
    raise RuntimeError(f"Failed to load model or tokenizer: {str(e)}")

##############################################################################
# Health check endpoint
##############################################################################
@app.route('/health-check', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200

##############################################################################
# Load ads dataset (fake endpoint)
##############################################################################
@app.route('/load-data', methods=['POST'])
def load_data():
    """Fake load-data endpoint."""
    try:
        return jsonify({"message": "No data loading required for this model."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

##############################################################################
# Run ad-pushing algorithm
##############################################################################
@app.route('/run-algo', methods=['POST'])
def run_algo():
    """Run the ad-pushing algorithm."""
    try:
        # Get query from request
        data = request.get_json()
        query = data.get("query", "").lower()

        if not query:
            raise ValueError("Query is empty.")

        test_input = f"Task: Match queries to ads.\nQuery: {query}\n\t"
        # input_ids = tokenizer(test_input, return_tensors="pt").input_ids.to(device)
        inputs = tokenizer(test_input, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        print("input id:", input_ids)
        print("attention mask:", attention_mask)
        

        # output
        output_ids = model.generate(
            input_ids,
            ##############################
            attention_mask=attention_mask,
            ##############################
            no_repeat_ngram_size=3,
            num_beams=20,
            max_length=300,
            pad_token_id=tokenizer.pad_token_id
        )

        print("output ids:", output_ids)

        generated_ad = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("generated ad:", generated_ad)
        return jsonify({"matched_ad": generated_ad}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

##############################################################################
# Run the Flask app
##############################################################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
