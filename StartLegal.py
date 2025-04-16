import os
import json

import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from streamlit.logger import get_logger
from chains import (
    load_embedding_model,
    load_llm,
)

# load api key lib
from dotenv import load_dotenv


logger = get_logger(__name__)


def init():
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

    st.session_state.documents_list = ['CNH', 'Comprovante de Residência', 'Certidão de Casamento']
    st.session_state.documents = []


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


agents = dict()

if 'init' not in st.session_state:
    st.session_state.init = True
    load_dotenv(".env")
    init()

st.set_page_config(page_title="StartLegal")

st.header("StartLegal - Módulo Revisor 📄", divider='gray')

st.subheader(
    "Anexe a minuta de uma escritura e em seguida os documentos necessários para revisão."
)

with st.sidebar:
    st.title("Etapas de Revisão")
#     st.text("Documentos apresentados:")
#     for doc in st.session_state.documents:
#         st.text(doc)

#     st.text("Documentos apresentados...")

# tabs = st.tabs(st.session_state.documents_list)

# for tab, document in zip(tabs, st.session_state.documents_list):
#     with tab:
#         # upload a your files
#         uploaded_file = st.file_uploader(
#             "Suba o documento em algum desses formatos: PDF, png, jpeg, ou txt.", 
#             accept_multiple_files=False,
#             type=["png", "jpg", "jpeg", "pdf", "txt"],
#             key=document
#         )

#         if uploaded_file:
#             st.write("A IA irá coletar e validar as informações presentes...")

#             # Text extraction and embedding using OCR and LLM to build a QA RAG
#             query = st.session_state.prompts[document].get('latest')['input']
#             agent = RAG_agent_document_validator(document, uploaded_file)
#             answer = agent.invoke({'input': query})['answer']

#             stream_handler = StreamHandler(st.empty())
#             for token in answer:
#                 stream_handler.on_llm_new_token(token=token)
            
#             # Visualize data from Minuta document
#             st.write("Dados da Minuta (parte compradora)")

#             stream_handler = StreamHandler(st.empty())
#             for token in st.session_state.minuta_comprador:
#                 stream_handler.on_llm_new_token(token=token)

#             # Ask to LLM a table showing the Document data and Minuta data
#             st.write(f"Validando de {document} com os dados da Minuta.")
            
#             context = "Primeira tabela " + \
#                     answer + "| Segunda tabela " + \
#                     st.session_state.minuta_comprador

#             system_prompt = """ 
#             Você é um assistente que revisa documentos e precisa auxiliar o usuário que faz o trabalho manual 
#             de checar se dados que foram escritos na Minuta estão escritos da mesma forma que nos documentos de origem. 
#             O usuário fornecerá duas tabelas após o termo 'Contexto'.
#             Responda gerando uma tabela que compara apenas os dados dessas duas tabelas fornecidas.
#             Ignore diferenças de letras maiúsculas e minúsculas, ou que tenham símbolos '.', '-', ou '/'. 
#             """ + f" Contexto: {context} "
#             prompt = ChatPromptTemplate(
#                     [
#                         ("system", system_prompt),
#                         ("human", "{input}")
#                     ]
#                 )
            
#             chain = prompt | st.session_state.llm | StrOutputParser()

#             final_answer = chain.invoke("Compare apenas os dados do {document} os quais também estejam presentes na Minuta.")

#             stream_handler = StreamHandler(st.empty())
#             for token in final_answer:
#                 stream_handler.on_llm_new_token(token=token)