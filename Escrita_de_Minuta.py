import streamlit as st
from rag_utils.config import init
from rag_utils.pipeline import RAG_document_retrieval
import threading
import logging
import time


session_state_status_percent = 0

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


def write_text_agents_thread(uploaded_files, docs, prompts, embeddings, vectorstore_config, llm):
    global session_state_status_percent

    if uploaded_files:
        st.session_state.status = "Processando documentos anexados..."
        logging.info("Iniciando o processamento dos documentos.")
        
        session_state_status_percent = 0
        len_uploaded_files = len(uploaded_files)
        
        # Simulate processing each uploaded file
        for p, uploaded_file in enumerate(uploaded_files):
            st.session_state.status = f"Processando {uploaded_file.name}..."
            session_state_status_percent = (p+1) / len_uploaded_files
            logging.info(f"Processando {uploaded_file.name}...")
            logging.info(f"(Thread): Progresso {session_state_status_percent:.2%}")

            # Find the prompts for the current document
            for doc in docs:
                first_name = doc.split()[0].lower()
                if first_name in uploaded_file.name.lower():
                    logging.info(f"Prompts encontrados para {doc}: {prompts[doc].get('latest')['prompt']}")
                    break

            # Collect and structure data from Buyers 
            answer = RAG_document_retrieval(
                document=doc,
                file=uploaded_file,
                prompts=prompts,
                logger=logger,
                embeddings=embeddings,
                vectordb_config=vectorstore_config,
                llm=llm,
                ocr_params={
                    'pages': None,
                    'lang': 'por'
                }
            )

            logging.info(f"Resposta do RAG: {answer}")

        st.session_state.status = "Documentos processados com sucesso!"
        logging.info("Documentos processados com sucesso!")


def write_paragraph_button_callback(uploaded_files, container, documents_list=None):
    global session_state_status_percent
    
    logger.info(f"write_paragraph_button_callback Prompts loaded: {st.session_state.prompts.keys()}")
    thread = threading.Thread(
        target=write_text_agents_thread,
        args=(
            uploaded_files, 
            documents_list,
            st.session_state.prompts, 
            st.session_state.embeddings, 
            st.session_state.vectorstore_config, 
            st.session_state.llm
        ),
        daemon=True
    )
    thread.start()
    
    with container:
        bar = st.progress(0, text_ocr)
        while session_state_status_percent*100 < 100:
            time.sleep(0.5)
            bar.progress(session_state_status_percent, text_ocr)
            logging.info(f"Parte compradora: Progresso {session_state_status_percent:.2%}")
        bar.empty()
        thread.join()

    session_state_status_percent = 0
    st.session_state.status = "Processamento finalizado!"
    logging.info("Parte compradora: Processamento finalizado!")


def build_container_files_uploader_and_text_writer(container, labels: dict, key, callback, documents_list=None):
    
    container.markdown(f"**{labels['markdown_label']}**")
    
    uploaded_files = container.file_uploader(
        labels['file_uploader_label'],
        type=["pdf", "jpg", "jpeg", "png"],
        key=f"{key}_file_uploader",
        accept_multiple_files=True
    )
    
    write_text_button = container.button(
        labels['button_label'],
        help="Clique para gerar o parágrafo com as informações extraídas dos documentos.",
        disabled=not uploaded_files,
        on_click=callback,
        args=(uploaded_files, container, documents_list),
        key=f"{key}_button"
    )
    
    if uploaded_files and write_text_button:
        container.write(f"Status: {st.session_state.status}")


if 'init_escrita_de_minuta_page' not in st.session_state:
    st.session_state.init_escrita_de_minuta_page = True
    if 'status' not in st.session_state:
        st.session_state.status = "Aguardando o upload dos documentos..."
    
    init()

    st.session_state.buyer_documents_list = [
        'CNH Comprador', 
        'Comprovante de Residência Comprador', 
        'Certidão de Casamento Comprador',
        'Pacto Antenupcial ou Declaração de União Estável',
        'CNH Cônjuge',
        'Quitação ITBI'
    ]
    
    st.session_state.owner_documents_list = [
        'CNPJ Vendedor',
        'CNH Vendedor',
        'Comprovante de Residência Vendedor'
    ]

    st.session_state.propery_documents_list = [
        'Matrícula do Imóvel',
    ]

text_ocr = "Extraindo informações dos documentos..."

st.title(body='✍️ StartLegal - Escritor de Minutas')
st.header("Assistente de Elaboração de Escrituras", divider='gray', )

st.write(
    "Anexe os documentos necessários das partes compradora e vendedora e a escritura do imóvel."
)

parte_compradora = st.container()

build_container_files_uploader_and_text_writer(
    container=parte_compradora,
    labels={
        'markdown_label': '**Parte Compradora**',
        'file_uploader_label': 'Anexe os documentos da parte compradora',
        'button_label': 'Gerar Parágrafo',
        'progress_text': text_ocr
    },
    key='parte_compradora',
    callback=write_paragraph_button_callback,
    documents_list=st.session_state.buyer_documents_list
)

st.divider()

parte_vendedora = st.container()

build_container_files_uploader_and_text_writer(
    container=parte_vendedora,
    labels={
        'markdown_label': '**Parte Vendedora**',
        'file_uploader_label': 'Anexe os documentos da parte vendedora',
        'button_label': 'Gerar Parágrafo',
        'progress_text': text_ocr
    },
    key='parte_vendedora',
    callback=write_paragraph_button_callback,
    documents_list=st.session_state.owner_documents_list
)

st.divider()

imovel = st.container()

build_container_files_uploader_and_text_writer(
    container=imovel,
    labels={
        'markdown_label': '**Escritura do Imóvel**',
        'file_uploader_label': 'Anexe a escritura do imóvel',
        'button_label': 'Gerar Parágrafo',
        'progress_text': text_ocr
    },
    key='imovel',
    callback=write_paragraph_button_callback,
    documents_list=st.session_state.propery_documents_list
)
