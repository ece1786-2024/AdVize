import streamlit as st

st.title("AdVize")

st.markdown("AdVize is a tool for app builders to more efficiently evaluate their ad-pushing algorithms.")

st.markdown("""
### Instructions:
1. Use the first uploader to select the Python file (`.py`) that defines the ad-pushing algorithm to be tested. The algorithm should meet the following specifications:
    * Function name: `push_matched_ad()`
    * Input parameters: 
        * `query` (String): a search query from an app user
        * `ad_database` (pandas DataFrame): a collection of commercial ads from different vendors
    * Returned value:
        * `matched_ad` (String): a advertisement that best matches the search query
2. Use the second uploader to select a CSV file (`.csv`) as the database of commercial ads.
3. Click the "Run" button to view the expected satisfaction rate if the selected ad-pushing algorithm is used in your search engine.
""")

st.markdown("""
### Uploaders:
""")

# File upload for the ad-pushing mechanism:
function_file = st.file_uploader("Select an ad-pushing algorithm:", type="py")
if function_file:
    st.write(f"Selected Function File: {function_file.name}")

# File upload for the ad database:
csv_file = st.file_uploader("Select a commercial ad database:", type="csv")
if csv_file:
    st.write(f"Selected CSV File: {csv_file.name}")

# Button for displaying the results:
run_clicked = st.button("Run")
st.markdown("### Result:")

if run_clicked:
    if function_file and csv_file:
        try:
            # TODO: Trigger the backend logic here.


            satisfaction_rate = 0
            st.success("Files processed successfully!")
            st.markdown(f"Satisfaction Rate: **{satisfaction_rate:.2f}%**")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Please upload both files before proceeding.")
else:
    st.markdown("Please click the \"Run button\" to see the results.")