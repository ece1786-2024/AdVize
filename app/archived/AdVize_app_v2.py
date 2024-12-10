'''
    FRONTEND UI:
'''

import streamlit as st
import requests

st.title("AdVize")

st.markdown("AdVize is a tool for app builders to more efficiently evaluate their ad-pushing algorithms.")

# """
# ### Instructions:
# 1. Use the first uploader to select the Docker image (`.tar`) that defines the ad-pushing algorithm to be tested.
#     * The input to the algorithm is a search query (String).
#     * The output of the algorithm is an advertisement (String).
# 2. (Optional) Use the second uploader to select a CSV file (`.csv`) for the database of commercial ads.
# 3. (Optional) Use the third uploader to select a CSV file (`.csv`) for the persona data.
# 4. Click the "Run" button to view the expected satisfaction rate if the selected ad-pushing algorithm is used in your search engine.
# """

st.markdown("""
### Instructions:
1. Use the first uploader to select the Docker image (`.tar`) that defines the ad-pushing algorithm to be tested.
    * The input to the algorithm is a search query (String).
    * The output of the algorithm is an advertisement (String).
2. Click the "Run" button to view the expected satisfaction rate if the selected ad-pushing algorithm is used in your search engine.
""")

st.markdown("""
### Uploaders:
""")

# File upload for the ad-pushing algorithm:
algo_file = st.file_uploader("Upload an ad-pushing algorithm:", type=["tar", "docker"])
if algo_file:
    st.write(f"Selected Docker Image File: {algo_file.name}")

# # File upload for the ad database:
# csv_file = st.file_uploader("Upload a commercial-ad database (Optional):", type="csv")
# if csv_file:
#     print(csv_file)
#     st.write(f"Selected CSV File: {csv_file.name}")

# # File upload for the persona data:
# persona_file = st.file_uploader("Upload persona data (Optional):", type="csv")
# if persona_file:
#     st.write(f"Selected CSV File: {persona_file.name}")

# Button for displaying the results:
run_clicked = st.button("Run")
st.markdown("### Result:")

# Make sure the port number matches:
API_UPLOAD_URL = "http://localhost:5001/upload"
API_UPLOAD_AND_RUN = "http://localhost:5001/upload-and-run"

if run_clicked:
    if algo_file:
        try:
            # First do file upload:
            files = {
                "docker_image": algo_file
            }
            response = requests.post(API_UPLOAD_AND_RUN, files=files)

            # Run algo on success:
            if response.status_code == 200:
                st.success("Files uploaded successfully!")
                response_data = response.json()
                st.write(response_data)
            else:
                st.error(f"Error uploading files: {response.json().get('error', 'Unknown error')}")

            satisfaction_rate = 0
            st.success("Files processed successfully!")
            st.markdown(f"Satisfaction Rate: **{satisfaction_rate:.2f}%**")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Please upload both files before proceeding.")
else:
    st.markdown("Please click the \"Run button\" to see the results.")