import streamlit as st  # used to create ui
import os
from langchain.embeddings import OpenAIEmbeddings # use for text embedding
from langchain.vectorstores import FAISS # for similarity search
from dotenv import load_dotenv      # load variable from .env
load_dotenv()
os.environ["OPEN_API_KEY"]= "OPEN_API_KEY"

# customizing web application using streamlit functions
st.set_page_config(page_title=" Kids Eduction",page_icon=":robot:")
st.header("hey , I can help you find similar things")

# initializing the object
embeddings= OpenAIEmbeddings()

# use to import csv file
from langchain.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path='my_Data.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['Words']
})


data=loader.load()
print(data)

db=FAISS.from_documents(data, embeddings)

def get_text():
    input_text = st.text_input("You: ", key= input)
    return input_text


user_input=get_text()
submit = st.button('Find similar Things')  

if submit:
    
    #If the button is clicked, the below snippet will fetch us the similar text
    docs = db.similarity_search(user_input)
    print(docs)
    st.subheader("Top Matches:")
    st.text(docs[0].page_content)
    st.text(docs[1].page_content)

