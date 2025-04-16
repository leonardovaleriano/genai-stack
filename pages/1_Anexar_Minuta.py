import streamlit as st
import logging
from streamlit.logger import get_logger
from rag_utils.config import init
from rag_utils.content_indexing import document_encoder_retriever
from rag_utils.document_retrieval import build_agent

logging.basicConfig(level = logging.INFO)

logger = get_logger(__name__)

if 'init' not in st.session_state:
    st.session_state.init = True
    init()
    
st.set_page_config(page_title="StartLegal - Anexar a Minuta")

st.sidebar.header("Dados obtidos da Minuta")

st.subheader(
    "Anexe a minuta da escritura para iniciar a revisão.",
    divider='gray'
)

# upload a your files
uploaded_file_minuta = st.file_uploader(
    "Suba o documento da Minuta em formato PDF.", 
    accept_multiple_files=False,
    type="pdf",
    key='minuta'
)

if uploaded_file_minuta:

    if 'rag_minuta' not in st.session_state:
        st.write("A IA irá coletar as informações presentes no documento...")
        st.session_state['rag_minuta'] = True

        minuta_retriever = document_encoder_retriever(
            document_name='Minuta', 
            uploaded_file=uploaded_file_minuta,
            ocr_params={
                'pages': [0],
                'lang': 'por'
            }, 
            logger=logger, 
            embeddings=st.session_state.embeddings,
            vectorstore_config=st.session_state.vectorstore_config
        )

        st.session_state.minuta_db = minuta_retriever

        minuta_system = st.session_state.prompts['Minuta Comprador'].get('latest').get('prompt_minuta', None)
        
        minuta_agent = build_agent(
            prompt=minuta_system, 
            vectorstore=minuta_retriever,
            logger=logger,
            history_context="",
            llm=st.session_state.llm
        )

        query = st.session_state.prompts['Minuta Comprador'].get('latest')['input_minuta']
        logger.info(f"{query}")
        
        minuta_response = minuta_agent.invoke({'input': query })
        
        answer = minuta_response['answer']
        
        st.session_state.minuta_comprador = answer

        minuta_system_owner = st.session_state.prompts['Minuta Vendedor'].get('latest').get('prompt_minuta', None)
        
        minuta_agent_owner = build_agent(
            prompt=minuta_system_owner, 
            vectorstore=minuta_retriever,
            logger=logger,
            history_context="",
            llm=st.session_state.llm
        )

        query_owner = st.session_state.prompts['Minuta Vendedor'].get('latest')['input_minuta']
        
        minuta_response_owner = minuta_agent_owner.invoke({'input': query_owner })
        
        answer_owner = minuta_response_owner['answer']
        
        st.session_state.minuta_vendedor = answer_owner

if 'minuta_comprador' in st.session_state:
    st.write(st.session_state.minuta_comprador)

if 'minuta_vendedor' in st.session_state:
    st.write(st.session_state.minuta_vendedor)