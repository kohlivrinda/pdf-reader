import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from templates import *

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
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-base")
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl" , model_kwargs={"temperature": 0.5 , "max_length": 1024})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever= vectorstore.as_retriever(), 
        memory = memory

    )
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i , message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title= "PDF CHATterjee", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("PDF CHATterjee :books: ")
    user_question = st.text_input("Ask question about your documents:")
    if user_question:
        handle_user_input(user_question)

    # st.write(user_template.replace("{{MSG}}", "hello, shawty!"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "hello, friend!"), unsafe_allow_html=True)

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
                st.write(text_chunks)
                
                # 3. create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                # 4. create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


 
    
if __name__ == '__main__':
    main()