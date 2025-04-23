from rag_utils.content_indexing import document_encoder_retriever
from rag_utils.qa_document_retrieval import build_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def RAG_document_retrieval(
    document, 
    file,
    prompts, 
    logger, 
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
    answer = agent.invoke({'input': query})['answer']
    
    return answer


def RAG_document_validator(document, document_answer, minuta_answer, llm):
    
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
    answer = chain.invoke(f"Compare apenas os dados do {document} os quais também estejam presentes na Minuta.")
    return answer