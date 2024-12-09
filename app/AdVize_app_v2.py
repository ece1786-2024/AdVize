'''
    FRONTEND UI:
'''

import streamlit as st
import requests

st.title("AdVize")

st.markdown("AdVize is a tool for app builders to more efficiently evaluate their ad-pushing algorithms.")

st.markdown("""
### Instructions:
1. Use the first uploader to enter the Docker image name that defines the ad-pushing algorithm to be tested. The provided Docker image should expose the following three endpoints:
    * `/health-check`: for verifying the Docker container and its underlying application runs correctly, ensuring readiness before processing any further requests
    * `/run-algo`: for processing incoming queries and returns the matched advertisements
    * `/load-data`: for loading the required dataset into the Docker container
2. Click the "Run" button to view the expected satisfaction rate if the selected ad-pushing algorithm is used in your search engine.
""")

st.markdown("""
### Inputs:
""")

# Text input for the Docker image name:
docker_image_name = st.text_input("Enter the name of the target Docker image:", placeholder="default-ad-pushing-algorithm")

# Button for displaying the results:
run_clicked = st.button("Run")
st.markdown("### Result:")

# Make sure the port number matches:
API_UPLOAD_URL = "http://localhost:5001/upload"
API_UPLOAD_AND_RUN = "http://localhost:5001/upload-and-run"

if run_clicked:
    if docker_image_name:
        try:
            # First do file upload:
            data = {
                "docker_image": docker_image_name
            }
            response = requests.post(API_UPLOAD_AND_RUN, json=data)

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