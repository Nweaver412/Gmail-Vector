import re
import os
import logging
import json
import lancedb

import pandas as pd
import numpy as np
import streamlit as st

from dotenv import load_dotenv
from bs4 import BeautifulSoup

from langchain.vectorstores import LanceDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.callbacks import StreamlitCallbackHandler

load_dotenv()

logging.basicConfig(level=logging.INFO)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
df = pd.read_csv('in/tables/parts.csv')
    
if "messages" not in st.session_state:
    st.session_state.messages = []
    ai_intro = "Hello, I'm Kai, your AI SQL Bot. I'm here to assist you with SQL queries.What can I do for you?"
    
    st.session_state.messages.append({"role":"assistant", "content" : ai_intro})

def simple_text_splitter(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        # If not the first chunk, start earlier to create an overlap
        if start > 0 and (start - overlap) > 0:
            start -= overlap
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        chunks.append(text[start:end])
        start += chunk_size
    return chunks

docs = df['bodyData'].tolist()

chunk_size = 1000
overlap = 200

document_chunks = [chunk for doc in docs for chunk in simple_text_splitter(doc, chunk_size, overlap)]

embeddings = OpenAIEmbeddings()
vectorstore = LanceDB.from_documents(documents=document_chunks, embedding=embeddings)

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever()
)

user_input = st.chat_input("Ask a question...")

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
    response = qa.invoke(user_input)

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


# return qa.invoke(query)