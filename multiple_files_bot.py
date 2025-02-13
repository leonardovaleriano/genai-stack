import os

import streamlit as st
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import ConcurrentLoader
from streamlit.logger import get_logger
from chains import (
    load_embedding_model,
    load_llm,
)

# load api key lib
from dotenv import load_dotenv

load_dotenv(".env")


url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)


embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})


def main():
    st.header("📄 Revise a Minuta da Escritura")

    # upload a your files
    uploaded_files = st.file_uploader(
			"Arraste os documentos necessários e a minuta da Escritura (PDF, png, jpeg, ou arquivos .txt)", 
			accept_multiple_files=True,
			type=["png", "jpg", "jpeg", "pdf", "txt"]
	  	     )

    text = ""
    for file in uploaded_files:
        bytes_data = file.read()
        file_format = file.name.split('.')[1].lower()
        text += f"NOME DO ARQUIVO: {file.name}"

        match file_format:
            case 'pdf':
                try:
                    pdf_reader = PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                except:
                    images = convert_from_bytes(bytes_data)
                    for i, image in enumerate(images):
                        # image.save(file.name.split('.')[0] + '.png')
                        text += pytesseract.image_to_string(image, lang='por')
            case 'txt':
                #with open(file, encoding='utf8', mode='r') as f:
                for line in file:
                    text += file.read()
                    st.write(file.read())
            case 'png':
                text += pytesseract.image_to_string(Image.open(file), lang='por')
            case 'jpg':
                text += pytesseract.image_to_string(Image.open(file), lang='por')
            case 'jpeg':
                text += pytesseract.image_to_string(Image.open(file), lang='por')
            case _:
                st.write("Formato do arquivo:", file.name, "não é suportado!")

    # langchain_textspliter
    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=200, length_function=len
                    )
    chunks = text_splitter.split_text(text=text)

    # Store the chuncks part in db (vector)
    vectorstore = Neo4jVector.from_texts(
        chunks,
        url=url,
        username=username,
        password=password,
        embedding=embeddings,
        index_name="multiple_files_bot",
        node_label="MultipleFilesBotChunk",
        pre_delete_collection=True, # Delete existing data in collection
    )
    questions_and_answers = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
    )

    # Accept user questions/query
    query = st.text_input(
        """
        Faça a revisão das cláusulas da Minuta pedindo para a IA extrair os dados dos documentos em anexo e 
        comparar com os dados da Minuta.
        """
    )

    if query:
        stream_handler = StreamHandler(st.empty())
        questions_and_answers.run(query, callbacks=[stream_handler])

if __name__ == "__main__":
    main()
