from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
import logging


def document_encoder_retriever(
        document_name: str, 
        uploaded_file,
        ocr_params: dict,
        logger: logging.Logger, 
        vectorstore_config: dict, 
        embeddings
):
    '''
    Indexing Phase:
        Documents are transformed into vector representations using dense embeddings.
        These vectors are stored in a vector database.
    '''

    ocr_pages = ocr_params.get('pages', None)
    ocr_lang = ocr_params.get('lang', None)
    
    if uploaded_file:
        bytes_data = uploaded_file.read()
        file_format = uploaded_file.name.split('.')[1].lower()
        
        # Extract text from document
        if ocr_lang and type(ocr_lang) == str:
            text = documents_OCR(
                uploaded_file, 
                logger, 
                bytes_data, 
                file_format, 
                pages=ocr_pages, 
                lang=ocr_lang
            )    
        else:
            text = documents_OCR(uploaded_file, logger, bytes_data, file_format)
        
        # langchain_textspliter
        chunks = text_chunking(document_name, text)

        # Store the chuncks part in db (vector)
        vectorstore = build_vectorstore(document_name, vectorstore_config, embeddings, chunks)

        return vectorstore


def documents_OCR(uploaded_file, logger, bytes_data, file_format, pages=None, lang='por'):
    '''
    OCR Step:
        Extract text from PDFs, images and txt files. 
    '''
    text = ""

    match file_format:
        case 'pdf':
            images = convert_from_bytes(bytes_data)

            if not pages:
                pages = list(range(len(images)))

            for i, image in enumerate(images):
                if i not in pages:
                    continue

                text += f"Página: {i} \n\n" + pytesseract.image_to_string(image, lang=lang)

        case 'txt':
            for line in uploaded_file:
                text += line

        case 'png':
            text += pytesseract.image_to_string(Image.open(uploaded_file), lang=lang)

        case 'jpg':
            text += pytesseract.image_to_string(Image.open(uploaded_file), lang=lang)

        case 'jpeg':
            text += pytesseract.image_to_string(Image.open(uploaded_file), lang=lang)

        case _:
            logger.error(f"Formato do arquivo: {uploaded_file.name} não é suportado!")
            
    return text


def text_chunking(document_name, text, size=10000, overlap=200, text_splitter=None):
    '''
    Chuncking Step:
        Split document content into smaller segments called chunks. 
        These can be paragraphs, sentences, or token-limited segments, making it easier for the model to search and retrieve only what's needed. 
        The chunking technique is crucial for optimizing RAG performance.
    '''
    if not text_splitter:
        text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=size,
                                chunk_overlap=overlap,
                                length_function=len, 
                                separators=['\n\n', '\n']
                        )
    
    chunks = text_splitter.split_text(text=text)
    chunks = [f"NOME_DO_DOCUMENTO: {document_name} " + chunk for chunk in chunks]
    return chunks


def build_vectorstore(reference_name, vectorstore_config, embeddings, chunks):
    '''
    Store Embeddings Step:
        Enconding all chunks as dense embeddings representation and store them in a Vector Database.
    '''
    vectorstore = Neo4jVector.from_texts(
            chunks,
            url=vectorstore_config['url'],
            username=vectorstore_config['username'],
            password=vectorstore_config['password'],
            embedding=embeddings,
            node_label=f"MultipleFilesBotChunk_{reference_name}",
            pre_delete_collection=True, # Delete existing data in collection
        )
    
    return vectorstore

