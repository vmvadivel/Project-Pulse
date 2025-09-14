import os
from dotenv import load_dotenv # New import!
load_dotenv() # New line!
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_groq import ChatGroq # Changed from langchain_openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Global variable to store the Qdrant client instance
qdrant_client_instance = None

# A simple Pydantic model to define the chat request body
class ChatRequest(BaseModel):
    query: str

# Create an instance of the FastAPI application
app = FastAPI()

# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to Project Pulse backend!"}

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingests a document to be added to the knowledge base.
    """
    global qdrant_client_instance
    
    try:
        temp_file_path = f"temp/{file.filename}"
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # Add this line to see if documents are being loaded
        print(f"Loaded {len(documents)} documents.")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        # Add this line to see if text chunks are being created
        print(f"Created {len(texts)} text chunks.")

        # Check if the list is empty before proceeding
        if not texts:
            return {"error": "No text could be extracted from the document."}
        
        #embeddings = OpenAIEmbeddings()
        embeddings = HuggingFaceEmbeddings()
        qdrant_client_instance = Qdrant.from_documents(
            texts,
            embeddings,
            location=":memory:",
            collection_name="project_documents",
            force_recreate_collection=True
        )
        
        os.remove(temp_file_path)
        
        return {
            "filename": file.filename,
            "message": "Document ingested and processed successfully."
        }
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

@app.post("/chat")
async def chat_with_docs(request: ChatRequest):
    """
    Handles user queries and provides answers based on ingested documents.
    """
    global qdrant_client_instance

    if qdrant_client_instance is None:
        return {"error": "Knowledge base not yet ingested. Please upload a document first."}

    retriever = qdrant_client_instance.as_retriever()

    template = """You are a helpful AI assistant for Project Pulse.
    Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Initialize the LLM with Groq
    #llm = ChatGroq(model="mixtral-8x7b-32768")
    llm = ChatGroq(model="gemma2-9b-it")

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(request.query)

    return {"response": response}   