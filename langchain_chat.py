import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
#from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
#import google.generativeai as genai
#from langchain.vectorstores import FAISS
#from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain.chat_models import ChatOpenAI
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

st.set_page_config(page_title="Hello Welcome to My chat APP using RAG and OPENAI", layout="wide")

st.markdown("""
## Document Genie: Get instant insights from RGPD Documents
This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging OPENAI model gpt-3.5-turbo. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

### How It Works

Follow these simple steps to interact with the chatbot:

1. **Enter Your API Key**: You'll need a OPENAI API key for the chatbot to access OPENAI models. Obtain your API key https://platform.openai.com/api-keys.

2. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

3. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")



# This is the first API key input; no need to repeat it in the main function.
#api_key = st.secrets["openai_api"]["openai_key"]
api_key = st.text_input("Enter your OPENAI API Key:", type="password", key="api_key_input")
api_key = st.secrets["openai_api"]["openai_key"]
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    api_key = st.secrets["openai_api"]["openai_key"]
    print(f"API Key --> {api_key}")
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, 
                       openai_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Load the model of choice
# def load_llm():
#     llm = CTransformers(
#         model="llama-2-7b-chat.ggmlv3.q8_0.bin",
#         model_type="llama",
#         max_new_tokens=512,
#         temperature=0.5
#     )
#     return llm
# llm = model
# chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())


# Function for conversational chat
def conversational_chat(query):
    chain = get_conversational_chain()
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]


def main():
    # Create a conversational chain
  
    st.header("Hello ! Ask me(gpt-3) about  🤗💁")

    #user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    # if user_question and api_key:  # Ensure API key and user question are provided
    #     user_input(user_question, api_key)
    ########################################
    # Initialize chat history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Done")



    

    # Initialize messages
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me(gpt-3) about " + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    # Create containers for chat history and user input
    response_container = st.container()
    container = st.container()

    # User input form
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to csv data 👉 (:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    # Display chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")


if __name__ == "__main__":
    main()