import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from streamlit_chat import message
from typing import List, Dict, Any

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings()
db = FAISS.load_local(
    "my_faiss_index", embeddings, allow_dangerous_deserialization=True
)

def run_llm(query, chat_history: List[Dict[str, Any]] = []):
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="gemma2-9b-It",
        temperature=0.5
    )
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_document_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rephrase_prompt = hub.pull('langchain-ai/chat-langchain-rephrase')
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=db.as_retriever(), prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=stuff_document_chain)
    result = qa.invoke(input={'input': query, "chat_history": chat_history})
    return result['answer']

st.markdown("""
    <style>
    .user-message {
        background-color: #DCF8C6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        max-width: 80%;
        float: right;
        clear: both;
        word-wrap: break-word;
    }
    .bot-message {
        background-color: #E8E8E8;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        max-width: 900%;
        float: left;
        clear: both;
        word-wrap: break-word;
    }
    .message-container {
        display: flex;
        flex-direction: column;
    }
    .message {
        display: flex;
        align-items: center;
    }
    .user-message img, .bot-message img {
        width: 30px;
        height: 30px;
        margin-right: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.header("Documentation Helper Bot")

prompt = st.text_input('Prompt', placeholder='Enter Your Prompt...')

# Initialize session state if not already present
if "chat_answer_history" not in st.session_state:
    st.session_state["chat_answer_history"] = []

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if prompt:
    with st.spinner("Generating Response"):
        generated_response = run_llm(query=prompt, chat_history=st.session_state['chat_history'])
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(generated_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response))

# Display the chat history
if st.session_state['chat_answer_history']:
    for i in range(len(st.session_state['chat_answer_history'])):
        st.markdown(f"""
        <div class="message-container">
            <div class="message user-message">
                <img src="https://img.icons8.com/color/48/000000/user-male-circle.png"/>
                <div>{st.session_state['user_prompt_history'][i]}</div>
            </div>
            <div class="message bot-message">
                <img src="https://img.icons8.com/?size=100&id=q7wteb2_yVxu&format=png&color=000000"/>
                <div>{st.session_state['chat_answer_history'][i]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
