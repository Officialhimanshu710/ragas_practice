# 📚 RAG Practice — Acids, Bases & Salts (Class 10 Chemistry)

A personal practice project to learn and remember the key concepts behind **Retrieval-Augmented Generation (RAG)**, **LLM evaluation with RAGAS**, and **serving AI chains as APIs with FastAPI + LangServe**.

This project builds a complete RAG pipeline over a CBSE Class 10 Chemistry PDF (Acids, Bases & Salts), evaluates its quality using RAGAS metrics, and exposes it as a REST API.

---

## 🗂️ Project Structure

```
ragas_practice/
│
├── chain.py          # Core RAG pipeline (shared by api.py and evaluate.py)
├── api.py            # FastAPI server — serves the RAG chain as an API
├── evaluate.py       # RAGAS evaluation — scores the RAG pipeline quality
│
├── cbse-class-10-science-notes-...pdf   # Source document
├── .env              # API keys (NOT pushed to GitHub)
├── .env.example      # Template for others to know what keys are needed
├── .gitignore        # Files excluded from GitHub
├── pyproject.toml    # Project dependencies (managed by uv)
└── README.md
```

---

## 🧠 Key Concept: What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique where instead of relying only on an LLM's training knowledge, you first **retrieve relevant context** from your own documents, and then **give that context to the LLM** to answer questions.

```
User Question
     │
     ▼
[Retriever] ──── searches ────► [Vector Store / ChromaDB]
     │                                    ▲
     │                             (PDF chunks stored
     │                              as embeddings)
     ▼
Relevant chunks (context)
     │
     ▼
[LLM (Groq/Llama)] ◄── Prompt: "Answer based on this context: ..."
     │
     ▼
Final Answer
```

**Why RAG?**
- LLMs have a knowledge cutoff and don't know your private documents
- RAG lets you "talk to your documents" accurately
- Reduces hallucinations because the answer comes from real source text

---

## 🔧 Libraries & Their Roles

### 🦜 LangChain
**The glue framework** — connects all components (LLM, retriever, prompt, output parser) into a single pipeline using the **LCEL (LangChain Expression Language)** syntax with the `|` (pipe) operator.

```python
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
```

| Component | Role |
|-----------|------|
| `PyPDFLoader` | Reads and parses the PDF file into LangChain `Document` objects |
| `RecursiveCharacterTextSplitter` | Splits long documents into smaller overlapping chunks so they fit in the LLM's context window |
| `ChatPromptTemplate` | Defines the system + user message structure sent to the LLM |
| `RunnablePassthrough` | Passes the user's question through the chain unchanged |
| `StrOutputParser` | Converts the LLM's response object into a plain Python string |

---

### 🤗 HuggingFace Embeddings (`sentence-transformers`)
**Converts text into numbers (vectors)** so that similar meaning = similar numbers.

- Model used: `sentence-transformers/all-MiniLM-L6-v2`
- When a user asks a question, it's converted to a vector and compared against stored chunk vectors to find the most relevant chunks

```
"What is pH?" ──► [0.23, -0.11, 0.87, ...] (384-dimensional vector)
```

---

### 🗄️ ChromaDB (`langchain-chroma`)
**The vector database** — stores all the text chunks as embedding vectors on disk/memory so they can be quickly searched.

- `Chroma.from_documents()` — stores your PDF chunks as vectors
- `.as_retriever()` — creates a retriever that fetches the top-k most relevant chunks for a given query

---

### ⚡ Groq + LLaMA 3.3 (`langchain-groq`)
**The LLM / brain of the system** — Groq is a fast inference provider. LLaMA 3.3 70B is the open-source model running on it.

- Takes the retrieved context + user question as input
- Generates a natural language answer

---

### 📊 RAGAS (`ragas`)
**Evaluates the quality of your RAG pipeline** — like a report card for your AI system.

RAGAS measures 4 key metrics:

| Metric | What it measures | Ideal Score |
|--------|-----------------|-------------|
| **Faithfulness** | Is the answer factually consistent with the retrieved context? (No hallucination) | 1.0 |
| **Answer Relevancy** | Is the answer actually relevant to the question asked? | 1.0 |
| **Context Precision** | Are the retrieved chunks actually useful for answering the question? | 1.0 |
| **Context Recall** | Did the retriever find ALL the information needed from the source? | 1.0 |

```python
result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
    llm=ragas_llm,
    embeddings=ragas_embedding
)
```

RAGAS itself uses an LLM (Groq) as the "judge" to score these metrics automatically.

---

### 🚀 FastAPI + LangServe (`fastapi`, `langserve`)
**Serves your RAG chain as a web API** so anyone can call it over HTTP.

- **FastAPI** — modern Python web framework, auto-generates interactive API docs at `/docs`
- **LangServe** — LangChain's tool to expose any chain as a REST API with a single line: `add_routes(app, rag_chain, path="/rag")`
- **Uvicorn** — the ASGI server that actually runs the FastAPI app

After running `api.py`, you get:

| Endpoint | What it does |
|----------|-------------|
| `http://localhost:8000/rag/playground/` | Interactive UI to test the chain in the browser |
| `http://localhost:8000/rag/invoke` | POST endpoint to call the chain programmatically |
| `http://localhost:8000/docs` | Full auto-generated API documentation |

---

### 🗃️ HuggingFace `datasets`
**Formats data for RAGAS evaluation** — RAGAS expects input in the HuggingFace `Dataset` format.

```python
dataset = Dataset.from_dict({
    "question": [...],
    "answer": [...],
    "contexts": [...],
    "ground_truths": [...],
    "reference": [...]
})
```

---

### 🔐 python-dotenv
**Loads secrets from a `.env` file** so API keys are never hardcoded in source code.

```python
load_dotenv()
Groq_API_KEY = os.getenv("Groq_Api_Key")
```

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/your-username/ragas_practice.git
cd ragas_practice
```

### 2. Set up the environment
```bash
# Install uv (if not already installed)
pip install uv

# Create virtual environment with Python 3.11
uv sync --python 3.11
```

### 3. Add your API key
```bash
# Copy the example file
cp .env.example .env
# Edit .env and add your Groq API key
# Get one free at: https://console.groq.com
```

### 4. Add your PDF
Place the PDF file in the project root folder (same folder as `chain.py`).

### 5. Run the API server
```bash
# On Windows (required to avoid Unicode errors in terminal)
$env:PYTHONUTF8=1; uv run api.py
```
Then open **http://localhost:8000/rag/playground/** in your browser.

### 6. Run RAGAS evaluation
```bash
uv run evaluate.py
```
Results are saved to `ragas_results.csv`.

---

## 🔁 How the Pipeline Works (Step by Step)

```
1. PDF Loaded
        │
        ▼
2. Split into chunks (1000 chars, 200 overlap)
        │
        ▼
3. Each chunk → Embedding vector (HuggingFace MiniLM)
        │
        ▼
4. Vectors stored in ChromaDB
        │
        ▼
5. User asks a question
        │
        ▼
6. Question → Embedding → Search ChromaDB → Top-k relevant chunks retrieved
        │
        ▼
7. Chunks + Question → Prompt → Groq LLaMA 3.3 70B
        │
        ▼
8. LLM generates answer based ONLY on retrieved context
        │
        ▼
9. (Optional) RAGAS scores the answer quality
```

---

## 📦 Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11 | Runtime |
| LangChain | >=1.2 | RAG framework |
| ChromaDB | >=1.5 | Vector store |
| HuggingFace sentence-transformers | >=5.4 | Text embeddings |
| Groq (LLaMA 3.3 70B) | via API | LLM |
| RAGAS | >=0.4.3 | RAG evaluation |
| FastAPI | >=0.136 | API server |
| LangServe | >=0.3.3 | LangChain to REST API |
| Uvicorn | >=0.44 | ASGI server |
| uv | latest | Python package manager |

---

## 📝 Personal Notes

- `chain.py` is the **shared core** — both the API and the evaluation import from it
- RAGAS needs the LLM wrapped with `LangchainLLMWrapper` to work with Groq
- Always run with `PYTHONUTF8=1` on Windows to avoid encoding issues with LangServe
- `sentence-transformers` and `scikit-network` require Python <= 3.13 on Windows (pre-built binaries not yet available for 3.14)
