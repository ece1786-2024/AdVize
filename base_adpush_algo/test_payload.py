import json

# Generate commercial ads data
test_data = {
    "query": "Nike Air Zoom Basketball Shoes",
    "commercial_ads_data": [
        {
            "search_query_bert_embedding": [0.05] * 768,
            "query_glove_embedding": [0.1] * 300,
            "Ads_Description": "Ad 1",
            "Search_Query": "Running Shoes"
        },
        {
            "search_query_bert_embedding": [0.1] * 768,
            "query_glove_embedding": [0.2] * 300,
            "Ads_Description": "Ad 2",
            "Search_Query": "Basketball Shoes"
        },
        {
            "search_query_bert_embedding": [0.15] * 768,
            "query_glove_embedding": [0.3] * 300,
            "Ads_Description": "Ad 3",
            "Search_Query": "Tennis Shoes"
        }
    ]
}

# Save JSON to a file
with open("test_payload.json", "w") as f:
    json.dump(test_data, f, indent=4)

print("Test payload written to test_payload.json")
