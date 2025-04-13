import streamlit as st

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
    st.write('Minuta Anexada!')