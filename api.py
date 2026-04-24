from fastapi import FastAPI
from langserve import add_routes
from chain import rag_chain  # import the shared RAG chain

app = FastAPI(
    title="RAG Chemistry API",
    description="Ask questions about acids, bases, and salts using a RAG pipeline"
)

add_routes(app, rag_chain, path="/rag")

if __name__ == "__main__":
    import uvicorn
    print("Starting server at http://localhost:8000")
    uvicorn.run(app, host="localhost", port=8000)