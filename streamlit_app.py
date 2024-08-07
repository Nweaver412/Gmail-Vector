import os
import logging
import openai
import lance
import zipfile
import streamlit as st
import pandas as pd
import numpy as np

from keboola.component import CommonInterface

from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain.callbacks import StreamlitCallbackHandler

from dotenv import load_dotenv

# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Keboola Input
ci = CommonInterface()
input_files = ci.get_input_files_definitions(tags=['zipped_lance'], only_latest_files=True)

# Find latest path
first_file = input_files[0]
zip_path = first_file.full_path

extract_path = "out/files/"

# Unzip the Lance dataset
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Load the Lance dataset from the extracted files
ds = lance.dataset(extract_path)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    ai_intro = "Hello, I'm Kai, your AI Assistant. I'm here to help you with your questions. What can I do for you?"
    st.session_state.messages.append({"role": "assistant", "content": ai_intro})

# Create embedding model
embed_model = OpenAIEmbedding(model="text-embedding-3-large", embed_batch_size=100)


# Creates doc arr, each doc labeled with id, text from db text and embedding from db vector
# Doc id generated pseudo randomly from batch and indeces
documents = []
for i, batch in enumerate(ds.to_batches()):
    for j, row in enumerate(batch.to_pylist()):
        doc_id = str(i * len(batch) + j)
        documents.append(Document(doc_id=doc_id, text=row['text'], embedding=row['vector']))

# Vector Store
vector_store = LanceDBVectorStore(
    uri="./lancedb",
    mode="overwrite",
    query_type="hybrid",
    dimension=3072 
)

# Store
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Index
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context, 
    embed_model=embed_model
)

# Engine
query_engine = index.as_query_engine(embed_model=embed_model)

# Streamlit UI
st.title("Kai - Your AI Assistant")

user_input = st.chat_input("Ask a question")

if user_input:
    # Add user message to the chat
    with st.chat_message("user"):
        st.markdown(user_input)
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display "Kai is typing..."
    with st.chat_message("Kai"):
        st.markdown("typing")
    st_callback = StreamlitCallbackHandler(st.container())
    response = query_engine.query(user_input)

    # Add Kai's message to session state
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Display Kai's message
    with st.chat_message("Kai"):
        st.markdown(response)
        # Display source nodes for Kai's response
        #st.write(response.source_nodes)

with st.container():    
    last_output_message = []
    last_user_message = []

    for message in reversed(st.session_state.messages):
        if message["role"] == "Kai":
            last_output_message = message
            break
    for message in reversed(st.session_state.messages):
        if message["role"] =="user":
            last_user_message = message
            break  