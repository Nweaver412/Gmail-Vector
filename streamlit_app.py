import os
import logging
import streamlit as st
import pandas as pd
import numpy as np
import lancedb

from llama_index.core import SimpleDirectoryReader, Document, StorageContext
from llama_index.chat_engine import CondenseQuestionChatEngine
from llama_index.chat_engine.condense_question import ChatMessage
from langchain.callbacks import StreamlitCallbackHandler
from llama_index.vector_stores import LanceDBVectorStore
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.prompts import Prompt
from dotenv import load_dotenv

# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize LanceDB
path = "/data/in/tables/embedded-gmail.csv"
df = pd.read_csv(path)
documents = [Document(content=row['bodyData'], metadata=dict(row)) for index, row in df.iterrows()]

# Set up LanceDB vector store
vector_store = LanceDBVectorStore(
    uri="./lancedb",
    mode="overwrite",
    query_type="hybrid"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create and index documents
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Custom prompt for question condensing
custom_prompt = Prompt("""\
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation. 
<Chat History> 
{chat_history}
<Follow Up Message>
{question}
<Standalone question>
""")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    ai_intro = "Hello, I'm Kai, your AI SQL Bot. I'm here to assist you with SQL queries. What can I do for you?"
    st.session_state.messages.append({"role": "assistant", "content": ai_intro})

# Create index and query engine
query_engine = index.as_query_engine()

# Create chat engine
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    condense_question_prompt=custom_prompt,
    verbose=True
)

# Streamlit UI
user_input = st.chat_input("Ask a question")

if user_input:
    # Add user message to the chat
    with st.chat_message("user"):
        st.markdown(user_input)
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display "Kai is typing..."
    with st.chat_message("Kai"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Kai is typing...")
    
    # Generate response
    st_callback = StreamlitCallbackHandler(st.container())
    response = chat_engine.chat(user_input)
    
    # Add Kai's message to session state
    st.session_state.messages.append({"role": "assistant", "content": str(response)})
    
    # Display Kai's message
    message_placeholder.markdown(str(response))

# Display chat history
with st.container():    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# return qa.invoke(query)