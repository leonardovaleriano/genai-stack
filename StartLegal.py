import streamlit as st
from streamlit.logger import get_logger


logger = get_logger(__name__)

st.set_page_config(page_title="StartLegal", page_icon="📄")

st.title(body='📄 StartLegal')
st.header("Módulo Revisor de Escrituras", divider='gray', )

st.write(
    "Anexe a minuta de uma escritura e em seguida os documentos necessários para revisão."
)

doc_ = '''Siga os passos abaixo para revisar informações da Minuta:
1. No menu à esquerda, clique em "Anexar Minuta" para inserir uma minuta no sistema e iniciar o processo de revisão.
2. Em seguida clique em "Parte Compradora" e insira no sistema os documentos necessários em cada aba disponível (se necessário).
    
    2.1. Aguarde o sistema extrair as informações e realizar a comparação com a Minuta fornecida.

    2.2. Caso encontre alguma inconsistência, reportar o escrivão e finalizar o processo de revisão.

3. Por último, clique em "Parte Vendedora" e insira os documentos solicitados.

    3.1. Aguarde o sistema extrair as informações e realizar a comparação com a Minuta fornecida.

    3.2 Caso encontre alguma inconsistência, reportar o escrivão e finalizar o processo de revisão.
'''

st.markdown(
    doc_
)