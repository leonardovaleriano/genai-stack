from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
import logging


def build_agent(prompt, vectorstore, logger: logging.Logger, history_context="", llm=None):
    if not llm:
        logger.error("LLM is not available!")

        return None
    
    if not prompt:
        # st.session_state.prompts[document_name].get('latest')['prompt']
        prompt = "You are a user assistant. Answer the questions using only the context provided." 

    system_prompt = prompt + " Context: {context} " + history_context + " "

    chat_prompt = ChatPromptTemplate(
            [
                ("system", system_prompt),
                ("human", "{input}")
            ]
        )
    
    qa_chain = create_stuff_documents_chain(llm, chat_prompt)

    agent_document_retrieval = create_retrieval_chain(vectorstore.as_retriever(), qa_chain)
    
    return agent_document_retrieval
