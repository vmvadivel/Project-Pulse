from fastapi import FastAPI

# Create an instance of the FastAPI application
app = FastAPI()

# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to Project Pulse backend!"}