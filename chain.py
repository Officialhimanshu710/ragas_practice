import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

Groq_API_KEY = os.getenv("Groq_Api_Key")

# LLM
llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=Groq_API_KEY)

# Load PDF file (relative to this script's location)
file_path = os.path.join(os.path.dirname(__file__), "cbse-class-10-science-notes-chapter-2-acids-bases-and-salts.pdf")
loader = PyPDFLoader(file_path)
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ".", " "]
)
chunks = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Build the prompt
system_prompt = (
    "You are an expert in chemistry. Your task is to answer the user's questions based "
    "on the provided context about acids, bases, and salts. Always provide answers in a "
    "clear, concise, and educational manner, suitable for a Class 10 student. If the answer "
    "is not present in the context, politely state that the information is not available in "
    "the provided text.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# The shared RAG chain
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
