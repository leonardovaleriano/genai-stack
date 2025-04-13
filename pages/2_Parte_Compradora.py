import streamlit as st

st.session_state.buyer_documents_list = [
    'CNH', 
    'Comprovante de Residência', 
    'Certidão de Casamento'
]

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