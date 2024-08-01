import streamlit as st
from demo_interface import DemoInterface


@st.cache_resource
def create_model_instances():
    json_path = 'raw_knowledge.json'
    return DemoInterface(json_path)

demo_interface = create_model_instances()

st.title("RAGatouille: A Demo Interface for RAG Models")

# URL for the logo image
logo_path = "./university-of-michigan.svg"

st.image(logo_path, width=100) 

st.write("Enter a query to search through the documents:")

query = st.text_input('Search Query')

if query:
    results = demo_interface.ask(query)
    st.write(f"Found {len(results)} results:")
    for idx, result in enumerate(results):
        st.write(f"**Result {idx + 1}:** {result}")
else:
    st.write("Type a query above to start searching.")
