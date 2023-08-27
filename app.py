import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

def get_pdf_text(pdf_docs):
    # text = ""
    # for pdf in pdf_docs:
    #     pdf_reader = PdfReader(pdf)
    #     for page in pdf_reader.pages:
    #         text+= page.extract_text()
    # return text
    pdf_texts = [page.extract_text() for pdf in pdf_docs for page in PdfReader(pdf).pages]
    return ''.join(pdf_texts)

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000 ,
        chunk_overlap = 200,
        length_function = len
    )

    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vectorstore = FAISS.from_text(texts = text_chunks, embeddings = embeddings)

def main():
    load_dotenv()
    st.set_page_config(page_title= "PDF CHATterjee", page_icon=":books:")
    st.header("PDF CHATterjee :books: ")
    st.text_input("Ask question about your documents:")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # 1. get pdf text (raw corpus)

                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # 2. chunk the text
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)


                # 3. create vector store

 
    
if __name__ == '__main__':
    main()