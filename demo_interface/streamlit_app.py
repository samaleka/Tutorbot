import streamlit as st
from demo_interface import DemoInterface
st.set_page_config(page_title="Demo for RAG Models",page_icon="🤖")

@st.cache_resource
def create_model_instances():
    json_path = 'raw_knowledge.json'
    return DemoInterface(json_path)

demo_interface = create_model_instances()
logo_path = "./university-of-michigan.svg"

#st.title("RAGatouille")
st.image(logo_path, width=100)
st.title("Demo Interface for RAG Models")


#st.image(logo_path, width=100) 
st.write("___")
st.subheader('Query')
#st.write("Enter a query to search through the documents:")

query = st.text_input('Type a query below to start searching')

if query:
    results, ctx = demo_interface.ask(query)
    st.write("___")
    st.subheader('Results')
    st.write(f"Found {len([results])} results:")
    #st.write("___")
    for idx, result in enumerate([results]):
        st.write(f"Answer: **{result.capitalize()}**")
        st.write("___")
        with st.expander("Show retrieved context", expanded=False):
            st.write(ctx)
#else:
#    st.write("Type a query above to start searching.")
