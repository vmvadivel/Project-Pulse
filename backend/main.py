from fastapi import FastAPI, File, UploadFile

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
    try:
        # Later we would add the document processing
        # logic here (e.g., reading the file, chunking, embedding, etc.).
        # For now, lets confirm receipt of the file.
        contents = await file.read()
        file_size_kb = len(contents) / 1024

        return {
            "filename": file.filename,
            "message": "Document received successfully.",
            "file_size_kb": f"{file_size_kb:.2f} KB"
        }
    except Exception as e:
        return {"error": f"An error occurred: {e}"}