import streamlit as st
from streamlit.logger import get_logger
import logging
from langchain.callbacks.base import BaseCallbackHandler
from rag_utils.config import init
from rag_utils.content_indexing import document_encoder_retriever
from rag_utils.document_retrieval import build_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


logging.basicConfig(level = logging.INFO)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


if 'init' not in st.session_state:
    st.session_state.init = True
    init()

if 'init_buyer_review_page' not in st.session_state:
    st.session_state.init_buyer_review_page = True

    st.session_state.buyer_documents_list = [
        'CNH', 
        'Comprovante de Residência', 
        'Certidão de Casamento'
    ]
    st.session_state.final_answer = dict().fromkeys(st.session_state.buyer_documents_list)

logger = get_logger(__name__)

# Define a list of Documents at app init() method
tabs = st.tabs(st.session_state.buyer_documents_list)

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

            # Text extraction and embedding using OCR and LLM to build a QA RAG
            document_retriever = document_encoder_retriever(
                document_name=document, 
                uploaded_file=uploaded_file,
                ocr_params={
                    'pages': None, # All pages
                    'lang': 'por'
                }, 
                logger=logger, 
                embeddings=st.session_state.embeddings,
                vectorstore_config=st.session_state.vectorstore_config
            )

            # prepare prompt with instructions
            instructions = st.session_state.prompts[document].get('latest')['prompt']
            agent = build_agent(
                prompt=instructions, 
                vectorstore=document_retriever, 
                logger=logger, 
                llm=st.session_state.llm
            )

            query = st.session_state.prompts[document].get('latest')['input']
            answer = agent.invoke({'input': query})['answer']
            stream_handler = StreamHandler(st.empty())
            for token in answer:
                stream_handler.on_llm_new_token(token=token)

            # Ask to LLM a table showing the Document data and Minuta data
            st.write(f"Validando de {document} com os dados da Minuta.")
            
            context = "Primeira tabela " + \
                    answer + "| Segunda tabela " + \
                    st.session_state.minuta_comprador

            system_prompt = """ 
            Você é um assistente que revisa documentos e precisa auxiliar o usuário que faz o trabalho manual 
            de checar se dados que foram escritos na Minuta estão escritos da mesma forma que nos documentos de origem. 
            O usuário fornecerá duas tabelas após o termo 'Contexto'.
            Responda gerando uma tabela que compara apenas os dados dessas duas tabelas fornecidas.
            Ignore diferenças de letras maiúsculas e minúsculas, ou que tenham símbolos '.', '-', ou '/'. 
            """ + f" Contexto: {context} "
            prompt = ChatPromptTemplate(
                    [
                        ("system", system_prompt),
                        ("human", "{input}")
                    ]
                )
            
            chain = prompt | st.session_state.llm | StrOutputParser()

            final_answer = chain.invoke("Compare apenas os dados do {document} os quais também estejam presentes na Minuta.")
            st.session_state.final_answer[document] = final_answer

            stream_handler = StreamHandler(st.empty())
            for token in final_answer:
                stream_handler.on_llm_new_token(token=token)
        else:
            if st.session_state.final_answer[document]:
                st.write(st.session_state.final_answer[document])