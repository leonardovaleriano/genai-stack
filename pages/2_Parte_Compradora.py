import streamlit as st
from streamlit.logger import get_logger
import logging
from utils import StreamHandler
from rag_utils.config import init
from rag_utils.content_indexing import document_encoder_retriever
from rag_utils.qa_document_retrieval import build_agent
from rag_utils.pipeline import RAG_document_retrieval, RAG_document_validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


logging.basicConfig(level = logging.INFO)

if 'init' not in st.session_state:
    st.session_state.init = True
    init()

if 'init_buyer_review_page' not in st.session_state:
    st.session_state.init_buyer_review_page = True

    st.session_state.buyer_documents_list = [
        'CNH Comprador', 
        'Comprovante de Residência Comprador', 
        'Certidão de Casamento Comprador',
        'Pacto Antenupcial ou Declaração de União Estável',
        'CNH Cônjuge',
        'Quitação ITBI'
    ]
    st.session_state.buyer_documents_list_tab = [
        'CNH', 
        'Comprovante de Residência', 
        'Certidão de Casamento',
        'Pacto Antenupcial ou Declaração de União Estável',
        'CNH Cônjuge',
        'Quitação ITBI'
    ]
    st.session_state.final_answer = dict().fromkeys(st.session_state.buyer_documents_list)

logger = get_logger(__name__)

# Define a list of Documents at app init() method
tabs = st.tabs(st.session_state.buyer_documents_list_tab)

for tab, document in zip(tabs, st.session_state.buyer_documents_list):
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

            # Collect and structure data from Buyers 
            answer = RAG_document_retrieval(
                    document=document,
                    file=uploaded_file,
                    prompts=st.session_state.prompts,
                    logger=logger,
                    embeddings=st.session_state.embeddings,
                    vectordb_config=st.session_state.vectorstore_config,
                    llm=st.session_state.llm,
                    ocr_params={
                        'pages': None,
                        'lang': 'por'
                    }
                )
        
            stream_handler = StreamHandler(st.empty())
            for token in answer:
                stream_handler.on_llm_new_token(token=token)

            # Ask to LLM a table showing the Document data and Minuta data
            st.write(f"Validando de {document} com os dados da Minuta.")

            final_answer = RAG_document_validator(
                document=document,
                document_answer=answer,
                minuta_answer=st.session_state.minuta_comprador,
                llm=st.session_state.llm,
                logger=logger
            )
            
            st.session_state.final_answer[document] = final_answer

            stream_handler = StreamHandler(st.empty())
            for token in final_answer:
                stream_handler.on_llm_new_token(token=token)
        else:
            if st.session_state.final_answer[document]:
                st.write(st.session_state.final_answer[document])