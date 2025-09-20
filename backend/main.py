import os
import shutil
from typing import Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# New Imports for Hybrid Search
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check for the GROQ API key
if "GROQ_API_KEY" not in os.environ:
    raise ValueError("GROQ_API_KEY environment variable not found. Please set it.")

# Initialize Groq Langchain chat model
# mistral-saba-24b  
# llama-3.3-70b-versatile

llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize FastAPI
app = FastAPI()

# In-memory history for conversation
conversation_history: list = []

# Global variable for the QA chain
qa_chain: Any = None

class ChatRequest(BaseModel):
    query: str

class IngestResponse(BaseModel):
    message: str
    num_documents: int
    num_chunks: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """Ingests a file and creates a vector store."""
    print("Received a new file for ingestion.")
    try:
        # Create a temporary directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        temp_file_path = os.path.join("temp", file.filename)
        print(f"File saved to temporary path: {temp_file_path}")

        # Save the uploaded content to the temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Determine the loader based on file extension
        ext = os.path.splitext(file.filename)[1].lower()
        if ext == ".pdf":
            # Using UnstructuredPDFLoader for PDFs
            loader = UnstructuredPDFLoader(temp_file_path)
        else:
            # Using UnstructuredFileLoader for other file types
            loader = UnstructuredFileLoader(temp_file_path)

        print("Starting document loading...")
        documents = loader.load()
        print("Document loading complete.")
        print(f"Loaded {len(documents)} documents.")

        # Using RecursiveCharacterTextSplitter for better chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )

        print("Starting text chunking...")
        texts = text_splitter.split_documents(documents)
        print("Text chunking complete.")
        print(f"Created {len(texts)} text chunks.")

       
        # Create a Qdrant vector store from the texts
        # Create a Qdrant vector store from the texts
        qdrant_vectorstore = Qdrant.from_documents(
            documents=texts,
            embedding=embeddings,
            location=":memory:",  # Use this parameter for an in-memory database
            collection_name="my_documents"
        )

        # Create a ConversationalRetrievalChain
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=qdrant_vectorstore.as_retriever(search_kwargs={'k': 5}),
            return_source_documents=True
        )

        qdrant_retriever = qdrant_vectorstore.as_retriever(search_kwargs={'k': 5})
        
        # Create a BM25 retriever from the texts (for keyword search)
        bm25_retriever = BM25Retriever.from_documents(documents=texts)
        bm25_retriever.k = 5

        # Create an Ensemble Retriever to combine both searches
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, qdrant_retriever],
            weights=[0.5, 0.5]
        )

        # Create a ConversationalRetrievalChain using the new hybrid retriever
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=ensemble_retriever,
            return_source_documents=True
        )

        # Store the QA chain in a global variable or cache for access in chat endpoint
        global qa_chain
        qa_chain = qa

        return IngestResponse(
            message="File ingested and vector store created successfully.",
            num_documents=len(documents),
            num_chunks=len(texts)
        )

    except Exception as e:
        print(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Error during ingestion: {e}")
    finally:
        # Cleanup the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/chat")
def chat_with_docs(request: ChatRequest):
    """Answers a user's query based on the ingested documents."""
    global qa_chain
    if qa_chain is None:
        raise HTTPException(status_code=400, detail="No documents have been ingested yet. Please upload a file first.")

    try:
        # Use the global conversation history
        result = qa_chain.invoke({'question': request.query, 'chat_history': conversation_history})
        response = result['answer']

        # Update conversation history with the new question and answer
        conversation_history.append((request.query, response))

        return {"response": response}
    except Exception as e:
        print(f"Error during chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error during chat: {e}")