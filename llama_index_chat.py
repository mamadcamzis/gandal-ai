import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from PyPDF2 import PdfReader
import openai
import os


#OPENAI_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = st.secrets["openai_api"]["openai_key"]
st.header("Binevenue à notre assistant RGPD 💬📚")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Posez moi une question sur le RGPD, Je serai ravi de vous aider!"}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Chargement et Indexation des documents RGPD, Cette opération peut prendre plusieurs minutes. Merci de votre patience !"):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo",
                                                                   temperature=0.5,
                                                                    system_prompt="""Vous êtes un assistant pour les tâches de réponses aux questions d’un employé qui analyse des données à caractère personnel.
Vous ne répondez qu’en français. Utilisez les éléments de contexte récupérés pour répondre à la question. Si vous ne connaissez pas la réponse, dites simplement que vous ne la savez pas. 
Si la question comprend le sigle DCP, interprète le en tant que Données à Caractère Personnel.
Si la question comprend le sigle ML, interprète le en tant que Apprentissage Machine. 
Si la question ne porte pas sur un sujet relatif au RGPD, répondez que vous n’êtes pas missionné pour répondre à cette question.  (peut être trop restrictif ça …  mais j’aimerais que ça marche si on demande quel temps il fera demain ….)
Si les éléments de contexte ne comprennent aucun élément suffisamment précis, répondez que les éléments trouvés ne sont pas assez précis, proposez de reformuler la question et dites quel élément n’a pas été trouvé. Par exemple si une question porte sur les clients résiliés et que les éléments de contexte ne contiennent ni le qualificatif « résilié » ni un synonyme comme « inactif », répondez « Je n’ai trouvé aucun élément suffisamment précis pour répondre à cette question, la notion de résilié n’y est pas abordée ». Vous pourrez ensuite proposer une réponse plausible en commençant par « Cependant, voici une réponse plausible à vérifier par d’autres sources : ».
Proposez une réponse claire, complète et concise, dans un style professionnel. Puis complétez votre réponse en donnant l’extrait le plus pertinent issu des éléments des contexte : nom du document, page, et extrait.
Pas besoin d'une réponse dans une autre langue que le français."""))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index


index = load_data()
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Votre question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("En recherche..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history