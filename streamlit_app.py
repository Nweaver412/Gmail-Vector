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


ci = CommonInterface()
input_files = ci.get_input_files_definitions(tags=['zipped_lance'], only_latest_files=True)

first_file = input_files[0]
zip_path = first_file.full_path

extract_path = "out/files/"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Load the Lance dataset from the extracted files
ds = lance.dataset(extract_path)

# Custom prompt for question condensing
# custom_prompt = Prompt("""\
# Given a conversation (between Human and Assistant) and a follow up message from Human, \
# rewrite the message to be a standalone question that captures all relevant context \
# from the conversation. 
# <Chat History> 
# {chat_history}
# <Follow Up Message>
# {question}
# <Standalone question>
# """)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    ai_intro = "Hello, I'm Kai, your AI Assistant. I'm here to help you with your questions. What can I do for you?"
    st.session_state.messages.append({"role": "assistant", "content": ai_intro})

# Create embedding model
embed_model = OpenAIEmbedding(model="text-embedding-3-large", embed_batch_size=100)

documents = []

# Creates doc arr, each doc labeled with id, text from db text and embedding from db vector
# Doc id generated pseudo randomly from batch and indeces

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
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Kai is typing...")
    
    st_callback = StreamlitCallbackHandler(st.container())
    response = query_engine.query(user_input)

    # Add Kai's message to session state
    st.session_state.messages.append({"role": "assistant", "content": str(response)})
    # Display Kai's message
    message_placeholder.markdown(str(response))

# Display chat history
# with st.container():    
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])