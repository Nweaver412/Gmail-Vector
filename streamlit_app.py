import os
import logging
import openai
import lance
import streamlit as st
import pandas as pd
import numpy as np
import zipfile

# from keboola.component import CommonInterface

from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.chat_engine import CondenseQuestionChatEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain.callbacks import StreamlitCallbackHandler
from llama_index.prompts import Prompt
from dotenv import load_dotenv

# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Unzip the input file
zip_path = "in/files/embedded_lance.zip"
extract_path = "out/files/"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Load the Lance dataset from the extracted files
ds = lance.dataset(extract_path)

ci = CommonInterface()

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
    ai_intro = "Hello, I'm Kai, your AI Assistant. I'm here to help you with your questions. What can I do for you?"
    st.session_state.messages.append({"role": "assistant", "content": ai_intro})

# Create embedding model
embed_model = OpenAIEmbedding(model="text-embedding-3-large", embed_batch_size=100)

def re_embed_text(text):
    return embed_model.get_text_embedding(text)

# Create documents from Lance dataset and re-embed the text
documents = []
for i, batch in enumerate(ds.to_batches()):
    for j, row in enumerate(batch.to_pylist()):
        doc_id = str(i * len(batch) + j)
        new_embedding = re_embed_text(row['text'])
        documents.append(Document(doc_id=doc_id, text=row['text'], embedding=new_embedding))

# Create LanceDBVectorStore
vector_store = LanceDBVectorStore(
    uri="./lancedb",
    mode="overwrite",
    query_type="hybrid",
    dimension=3072
)

# Create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context, 
    embed_model=embed_model
)

# Create query engine
query_engine = index.as_query_engine(embed_model=embed_model)

# Create chat engine
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    condense_question_prompt=custom_prompt,
    verbose=True
)

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