import streamlit as st
from streamlit.logger import get_logger
import os
import json

from chains import (
    load_embedding_model,
    load_llm,
)


logger = get_logger(__name__)

# load api key lib
from dotenv import load_dotenv


def init():
    load_dotenv(".env")

    st.session_state.vectorstore_config = dict()
    st.session_state.vectorstore_config['url'] = os.getenv("NEO4J_URI")
    st.session_state.vectorstore_config['username'] = os.getenv("NEO4J_USERNAME")
    st.session_state.vectorstore_config['password'] = os.getenv("NEO4J_PASSWORD")
    
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    embedding_model_name = os.getenv("EMBEDDING_MODEL")
    llm_name = os.getenv("LLM")
    # Remapping for Langchain Neo4j integration
    os.environ["NEO4J_URL"] = st.session_state.vectorstore_config['url']

    embeddings, dimension = load_embedding_model(
        embedding_model_name, 
        config={"ollama_base_url": ollama_base_url}, 
        logger=logger
    )
    st.session_state.embeddings = embeddings
    st.session_state.dimension = dimension

    prompts = dict()
    with open('prompts.json', 'rb') as f:
        prompts = json.load(f)
    
    st.session_state.prompts = prompts
    st.session_state.llm = load_llm(
        llm_name, 
        logger=logger, 
        config={"ollama_base_url": ollama_base_url}
    )
