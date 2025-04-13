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
    st.session_state.url = os.getenv("NEO4J_URI")
    st.session_state.username = os.getenv("NEO4J_USERNAME")
    st.session_state.password = os.getenv("NEO4J_PASSWORD")
    
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    embedding_model_name = os.getenv("EMBEDDING_MODEL")
    llm_name = os.getenv("LLM")
    # Remapping for Langchain Neo4j integration
    os.environ["NEO4J_URL"] = st.session_state.url

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


def RAG_Minuta(document_name: str, uploaded_file):
    text = ""
    if uploaded_file:
        bytes_data = uploaded_file.read()
        file_format = uploaded_file.name.split('.')[1].lower()
        
        match file_format:
            case 'pdf':
                images = convert_from_bytes(bytes_data)
                # Somente a 1ª página
                text += pytesseract.image_to_string(images[0], lang='por') + " \n\n"

                # for i, image in enumerate(images):
                #     text += f"Página: {i} \n\n" + pytesseract.image_to_string(image, lang='por')
            case _:
                st.write("Formato do arquivo:", uploaded_file.name, "não é suportado!")

        # langchain_textspliter
        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=10000,
                            chunk_overlap=200,
                            length_function=len, 
                            separators=['\n\n', '\n']
                        )
        chunks = text_splitter.split_text(text=text)
        chunks = [f"NOME_DO_DOCUMENTO: {document_name} " + chunk for chunk in chunks]

        # Store the chuncks part in db (vector)
        vectorstore = Neo4jVector.from_texts(
            chunks,
            url=st.session_state.url,
            username=st.session_state.username,
            password=st.session_state.password,
            embedding=st.session_state.embeddings,
            node_label=f"MultipleFilesBotChunk_{document_name}",
            pre_delete_collection=True, # Delete existing data in collection
        )

        return vectorstore


def RAG_agent_document_validator(document_name: str, uploaded_file):
    text = ""
    if uploaded_file:
        bytes_data = uploaded_file.read()
        file_format = uploaded_file.name.split('.')[1].lower()
        
        match file_format:
            case 'pdf':
                images = convert_from_bytes(bytes_data)
                for i, image in enumerate(images):
                    text += f"Página: {i} \n\n" + pytesseract.image_to_string(image, lang='por')
            case 'txt':
                for line in uploaded_file:
                    text += line
            case 'png':
                text += pytesseract.image_to_string(Image.open(uploaded_file), lang='por')
            case 'jpg':
                text += pytesseract.image_to_string(Image.open(uploaded_file), lang='por')
            case 'jpeg':
                text += pytesseract.image_to_string(Image.open(uploaded_file), lang='por')
            case _:
                st.write("Formato do arquivo:", uploaded_file.name, "não é suportado!")

        # langchain_textspliter
        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=10000,
                            chunk_overlap=200,
                            length_function=len
                        )
        chunks = text_splitter.split_text(text=text)
        chunks = [f"NOME_DO_DOCUMENTO: {document_name} " + chunk for chunk in chunks]

        # Store the chuncks part in db (vector)
        vectorstore = Neo4jVector.from_texts(
            chunks,
            url=st.session_state.url,
            username=st.session_state.username,
            password=st.session_state.password,
            embedding=st.session_state.embeddings,
            node_label=f"MultipleFilesBotChunk_{document_name}",
            pre_delete_collection=True, # Delete existing data in collection
        )

        agent_document_retreiver = build_RAG_agent(document_name, vectorstore)

        return agent_document_retreiver


def build_RAG_agent(document_name, vectorstore, prompt=None, history_context=""):
    if not prompt:
        prompt = st.session_state.prompts[document_name].get('latest')['prompt']

    system_prompt = prompt + " Context: {context} " + history_context + " "
    prompt = ChatPromptTemplate(
            [
                ("system", system_prompt),
                ("human", "{input}")
            ]
        )
    qa_chain = create_stuff_documents_chain(st.session_state.llm, prompt)
    agent_document_retreiver = create_retrieval_chain(vectorstore.as_retriever(), qa_chain)
    return agent_document_retreiver


agents = dict()


def main():
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
        st.title("Partes Envolvidas")
        st.text("Documentos apresentados:")
        for doc in st.session_state.documents:
            st.text(doc)

        st.text("Documentos apresentados...")

    # upload a your files
    uploaded_file_minuta = st.file_uploader(
        "Suba o documento da Minuta em formato PDF.", 
        accept_multiple_files=False,
        type="pdf",
        key='minuta'
    )

    if uploaded_file_minuta:
        if not st.session_state.get('rag_minuta', False):   
            st.write("A IA irá coletar as informações presentes no documento...")
            
            minuta_vector_db = RAG_Minuta('Minuta', uploaded_file_minuta)
            if 'rag_minuta' not in st.session_state:
                st.session_state['rag_minuta'] = True
                st.session_state.minuta_db = minuta_vector_db

                # Print a table with Minuta information
                if 'minuta_db' in st.session_state:
                    minuta_system = st.session_state.prompts['Minuta Comprador'].get('latest').get('prompt_minuta', None)
                    minuta_agent = build_RAG_agent('Minuta', st.session_state.minuta_db, minuta_system) # " gere uma tabela juntando a coluna 'Valor' das tabelas existentes."
                    
                    query = st.session_state.prompts['Minuta Comprador'].get('latest')['input_minuta']
                    minuta_response = minuta_agent.invoke({'input': query })
                    answer = minuta_response['answer']
                    st.session_state.minuta_comprador = answer

                    minuta_system = st.session_state.prompts['Minuta Vendedor'].get('latest').get('prompt_minuta', None)
                    minuta_agent = build_RAG_agent('Minuta', st.session_state.minuta_db, minuta_system) # " gere uma tabela juntando a coluna 'Valor' das tabelas existentes."
                    
                    query = st.session_state.prompts['Minuta Vendedor'].get('latest')['input_minuta']
                    minuta_response = minuta_agent.invoke({'input': query })
                    answer = minuta_response['answer']
                    st.session_state.minuta_vendedor = answer

    # Activate tabs after Minuta has been processed...
    if st.session_state.minuta:
        tabs = st.tabs(st.session_state.documents_list)

        for tab, document in zip(tabs, st.session_state.documents_list):
            with tab:
                # upload a your files
                uploaded_file = st.file_uploader(
                    "Suba o documento em algum desses formatos: PDF, png, jpeg, ou txt.", 
                    accept_multiple_files=False,
                    type=["png", "jpg", "jpeg", "pdf", "txt"],
                    key=document
                )

                if uploaded_file:
                    st.write("A IA irá coletar e validar as informações presentes...")

                    # Text extraction and embedding using OCR and LLM to build a QA RAG
                    query = st.session_state.prompts[document].get('latest')['input']
                    agent = RAG_agent_document_validator(document, uploaded_file)
                    answer = agent.invoke({'input': query})['answer']

                    stream_handler = StreamHandler(st.empty())
                    for token in answer:
                        stream_handler.on_llm_new_token(token=token)
                    
                    # Visualize data from Minuta document
                    st.write("Dados da Minuta (parte compradora)")

                    stream_handler = StreamHandler(st.empty())
                    for token in st.session_state.minuta_comprador:
                        stream_handler.on_llm_new_token(token=token)

                    # Ask to LLM a table showing the Document data and Minuta data
                    st.write(f"Validando de {document} com os dados da Minuta.")
                    
                    context = "Primeira tabela " + \
                            answer + "| Segunda tabela " + \
                            st.session_state.minuta_comprador

                    system_prompt = """ 
                    Dentro do contexto existem duas tabelas. 
                    Compare os dados em comum e responda ao usuário através de uma 
                    tabela com uma coluna indicando se existe diferença de escrita 
                    entre a primeira e segunda tabelas. 
                    Ignore diferenças causadas por Letras Maiúsculas e Minúsculas, 
                    ou símbolos '.', '-', ou '/'.
                    """ + f" Contexto: {context} "
                    prompt = ChatPromptTemplate(
                            [
                                ("system", system_prompt),
                                ("human", "{input}")
                            ]
                        )
                    
                    chain = prompt | st.session_state.llm | StrOutputParser()

                    final_answer = chain.invoke("Valide os dados do {document} com os dados da Minuta.")

                    stream_handler = StreamHandler(st.empty())
                    for token in final_answer:
                        stream_handler.on_llm_new_token(token=token)


if __name__ == "__main__":
    main()
