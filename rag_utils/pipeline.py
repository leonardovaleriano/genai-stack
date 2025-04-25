from rag_utils.content_indexing import document_encoder_retriever
from rag_utils.qa_document_retrieval import build_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback
import logging


def RAG_document_retrieval(
    document, 
    file,
    prompts, 
    logger: logging.Logger, 
    embeddings, 
    vectordb_config,
    llm,
    ocr_params={'pages': None, 'lang': 'por'}
) -> str:
    # Text extraction and embedding using OCR and LLM to build a QA RAG
    document_retriever = document_encoder_retriever(
        document_name=document, 
        uploaded_file=file,
        ocr_params=ocr_params, 
        logger=logger, 
        embeddings=embeddings,
        vectorstore_config=vectordb_config
    )

    # prepare prompt with instructions
    instructions = prompts[document].get('latest')['prompt']
    agent = build_agent(
        prompt=instructions, 
        vectorstore=document_retriever, 
        logger=logger, 
        llm=llm
    )

    # QA RAG document retrieval
    query = prompts[document].get('latest')['input']

    with get_openai_callback() as cb:
        answer = agent.invoke({'input': query})['answer']

    logger.info(f"Total Tokens: {cb.total_tokens}")
    logger.info(f"Prompt Tokens: {cb.prompt_tokens}")
    logger.info(f"Completion Tokens: {cb.completion_tokens}")
    logger.info(f"Total Cost (USD): ${cb.total_cost}")
    
    return answer


def RAG_document_validator(document, document_answer, minuta_answer, llm, logger: logging.Logger):
    
    # Build context aggregating information from document and Minuta
    context = f"Tabela {document} " + \
        document_answer + "| Tabela Minuta" + \
        minuta_answer

    # Instructions of how to check if Minuta information matches document information
    system_prompt = """ 
    Você é um assistente que compara dados obtidos de diferentes documentos. 
    O usuário fornecerá duas tabelas após o termo 'Contexto'.
    Auxilie o usuário a checar se os dados nessas duas tabelas estão escritos da mesma forma. 
    A comparação dos dados em comum precisa estar numa tabela. 
    Dados que aparecem em apenas uma das tabelas fornecidas não precisa aparecer na tabela de comparação.
    A comparação pode ignorar diferenças entre letras maiúsculas e minúsculas, e a presença de símbolos '.', '-', ou '/'.
    A tabela de comparação precisa ter uma coluna 'Validação' que indica se os dados foram escritos de forma idêntica. 
    """ + f" Contexto: {context} "
    prompt = ChatPromptTemplate(
            [
                ("system", system_prompt),
                ("human", "{input}")
            ]
        )
    
    chain = prompt | llm | StrOutputParser()
    
    # QA RAG document validation
    with get_openai_callback() as cb:
        answer = chain.invoke(f"Compare apenas os dados do {document} os quais também estejam presentes na Minuta.")
        
    logger.info(f"Total Tokens: {cb.total_tokens}")
    logger.info(f"Prompt Tokens: {cb.prompt_tokens}")
    logger.info(f"Completion Tokens: {cb.completion_tokens}")
    logger.info(f"Total Cost (USD): ${cb.total_cost}")

    return answer