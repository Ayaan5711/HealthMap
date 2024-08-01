import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai

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

    pdf_files = [os.path.join("dataset2", file) for file in os.listdir("dataset2") if file.endswith(".pdf")]

    dataset_dir = "dataset2"
    
    # Check if the directory exists
    if not os.path.exists(dataset_dir):
        print(f"Directory '{dataset_dir}' does not exist. Creating it.")
        os.makedirs(dataset_dir)


    pdf_files = [os.path.join("dataset2", file) for file in os.listdir("dataset2") if file.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in the dataset directory.")
        return
    
    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    create_vector_store(text_chunks)

def get_conversational_chain(context, chat_history, question):

    prompt_template = f"""You are an Indian AI health assistant designed to provide general health information and suggestions based on current medical knowledge and best practices. When responding to a user's query, please follow these guidelines:

1. Provide general, widely accepted advice for managing common symptoms or conditions. Present each piece of advice on a new line for clarity.
2. Suggest basic home remedies or over-the-counter treatments when appropriate. Make sure to list each suggestion on a new line.
3. Avoid including disclaimers about being an AI or limitations on providing medical diagnoses or personalized treatment advice.
4. Advise users to consult with a healthcare professional if symptoms persist, worsen, or if they have any concerns. List these recommendations clearly.
5. Provide information about when to seek immediate medical attention for severe symptoms. Clearly outline the circumstances that warrant immediate attention.
6. Always suggest a medicine for the symptoms and ensure that it is available in India. List the medicine(s) with their availability.

Context: {context}

Chat History: {chat_history}

Question: {question}

Answer:
"""

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0.3, 
        system_instruction="You are Healy, a highly experienced healthcare professional providing medical advice based on current medical knowledge and best practices. You will respond to the user's queries by leveraging your medical expertise and the Context Provided."
    )

    messages = [
        ("system", "You are an Indian AI health assistant designed to provide general health information and suggestions based on current medical knowledge and best practices. You will respond to the user's queries by leveraging your medical knowledge and the Context Provided.1. Provide general, widely accepted advice for managing common symptoms or conditions.2. Suggest basic home remedies or over-the-counter treatments when appropriate.3. Do not include a disclaimer that you are an AI and cannot provide medical diagnosis or personalized treatment advice.4. Advise users to consult with a healthcare professional if symptoms persist, worsen, or if they have any concerns.5. Offer information about when to seek immediate medical attention for severe symptoms.6. Always suggest a medicine for the symptoms and make sure it is available in India."),  
        ("user", f"{question}"),  
        ("assistant","")  
    ]

    prompt = ChatPromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"],
        messages=messages
    )

    chain = LLMChain(llm=model, prompt=prompt)  
    
    return chain

async def user_input(user_question, chat_history):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("Faiss", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(context=docs, chat_history=chat_history, question=user_question)
    
    response = chain.run({
        "context": docs, 
        "chat_history": chat_history, 
        "question": user_question
     })
    
    return response
