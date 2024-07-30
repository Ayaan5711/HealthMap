import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("Faiss")

def ingest_data():
    pdf_files = [os.path.join("dataset", file) for file in os.listdir("dataset") if file.endswith(".pdf")]
    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    create_vector_store(text_chunks)

def get_conversational_chain():
    prompt_template = """
    You are an Indian AI health assistant designed to provide general health information and suggestions based on current medical knowledge and best practices. 
    You will respond to the user's queries by leveraging your medical knowledge and the Context Provided.
    1. Provide general, widely accepted advice for managing common symptoms or conditions.
    2. Suggest basic home remedies or over-the-counter treatments when appropriate.
    3. Do not include a disclaimer that you are an AI and cannot provide medical diagnosis or personalized treatment advice.
    4. Advise users to consult with a healthcare professional if symptoms persist, worsen, or if they have any concerns.
    5. Offer information about when to seek immediate medical attention for severe symptoms.
    6. Always suggest a medicine for the symptoms and make sure it is available in India.
    Context: {context}
    Chat History: {chat_history}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", 
        temperature=0.3, 
        system_instruction="You are Healy, a highly experienced healthcare professional providing medical advice based on current medical knowledge and best practices. You will respond to the user's queries by leveraging your medical expertise and the Context Provided.")
    prompt = ChatPromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, chat_history):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("Faiss", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "chat_history": chat_history, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    
    st.set_page_config("Health", page_icon=":hospital:")
    st.header("AI Health Assistant :hospital:")
    if "data_ingested" not in st.session_state:
        st.session_state.data_ingested = False

    if not st.session_state.data_ingested:
        st.write("Ingesting data, please wait...")
        ingest_data()
        st.session_state.data_ingested = True
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi I'm  AI Health Advisor. How can I assist you today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Type your health-related question here...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                    response = user_input(prompt, chat_history)
                    st.write(response)

            if response is not None:
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)

if __name__ == "__main__":
    main()