# 🛡️ Sentinel | Enterprise Policy Assistant System

A production-grade, secure **Enterprise Policy Assistant** powered by **Groq High-Speed Inference** and **LangChain**. Sentinel delivers accurate, citation-backed answers to workplace policy questions while enforcing enterprise safety controls, prompt-injection defense, PII redaction, and local offline embeddings—**requiring zero OpenAI keys or credits**.

---

## 🌟 Key Features

- ⚡ **Groq Ultra-Fast Inference**:
  - Out-of-the-box support for the **GPT 120** model (`openai/gpt-oss-120b`).
  - Supports **Llama 3.3 70B** (`llama-3.3-70b-versatile`), **Llama 3.1 8B**, and **Mixtral 8x7B**.
  - Intelligent alias resolution (e.g. `gpt-120`, `120b` automatically map to `openai/gpt-oss-120b`).
- 🔒 **Zero API-Key Local Embeddings**:
  - High-performance local ONNX embeddings via Chroma.
  - Generates 384-dimensional vector representations locally—no OpenAI key or paid embedding API required.
- 🔒 **Restricted Enterprise Q&A Mode**:
  - End-users can only query the verified policy library. Document/PDF uploading is disabled to ensure policy integrity.
- 🛡️ **Layered Trust-by-Design Guardrails**:
  - **Prompt Injection Defense**: Multi-tier detection using deterministic regex patterns and LLM classification.
  - **Automated PII Redaction**: Sanitizes sensitive personal data (Email, Phone, PAN, Aadhaar, Credit Cards) on both user inputs and assistant outputs.
  - **Evidence Groundedness Validation**: Evaluates whether factual claims in the generated response are backed by retrieved context.
- 📚 **Source-Level Traceability**:
  - Every material claim includes verifiable citations (`[S1]`, `[S2]`) with document name, page number, section, and excerpt.
- 📊 **Evaluation Lab & Red-Teaming**:
  - **Benchmark Suite**: Quantitative evaluation against a curated golden dataset measuring retrieval hit rate, groundedness, relevance, and correctness.
  - **Adversarial Red Team Suite**: Probes prompt injection resilience, sensitive data leakage, and out-of-scope boundaries.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
[ Input Guardrails ] ────► PII Redaction (Email, Phone, etc.)
    │                ────► Prompt Injection Detection (Regex + LLM)
    ▼
[ Semantic Retrieval ] ──► Chroma Vector Store (Local ONNX Embeddings)
    │                ──► Top-K Relevant Policy Chunks
    ▼
[ Evidence Synthesis ] ──► Groq Inference Engine (openai/gpt-oss-120b)
    │                ──► Context Isolation & Strict Citation Prompting
    ▼
[ Output Assurance ] ──► Groundedness Validation Assessment
    │                ──► Output PII Redaction Check
    ▼
Final Answer + Source Citations + Telemetry Signals
```

---

## 📁 Repository Structure

```
enterprise-policy-assistant-system/
├── app.py                     # Streamlit web application & interactive dashboard
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
├── .env                       # Active environment configuration
├── data/
│   ├── policies/              # Default enterprise policy documents (Markdown/PDF/DOCX)
│   │   ├── ai_use_policy.md
│   │   ├── hr_policy.md
│   │   ├── it_security_policy.md
│   │   └── travel_policy.md
│   ├── uploads/               # User-uploaded policy documents
│   └── evaluation/            # Golden and red-team evaluation datasets
├── chroma_db/                 # Persistent Chroma vector database
├── src/
│   ├── config.py              # Central settings and model alias normalization
│   ├── llm.py                 # Unified Groq chat model factory
│   ├── vector_store.py        # Local embeddings, vector store & retrieval
│   ├── document_loader.py     # Multi-format document parser and text splitter
│   ├── rag_pipeline.py        # Enterprise policy assistant RAG workflow
│   ├── guardrails.py          # Injection detection, PII redactor & groundedness judge
│   ├── evaluation.py          # Quality benchmarking and red-team probe runners
│   └── schemas.py             # Pydantic data schemas for structured outputs
└── tests/
    ├── conftest.py            # Test configuration & path setup
    └── test_model_config.py   # Unit test suite for models, embeddings & safety
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10 or higher
- A free [Groq API Key](https://console.groq.com/)

### 2. Installation
Clone the repository and install the dependencies:

```bash
git clone <repository-url>
cd enterprise-policy-assistant-system
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file from the provided template:

```bash
cp .env.example .env
```

Open `.env` and add your **Groq API Key**:

```dotenv
# LLM Configuration (Powered by Groq)
GROQ_API_KEY=gsk_your_groq_api_key_here

# Supported Groq Models:
#   - openai/gpt-oss-120b       (GPT 120 / GPT-OSS 120B on Groq - Default)
#   - llama-3.3-70b-versatile   (Meta Llama 3.3 70B)
#   - llama-3.1-8b-instant      (Fast lightweight model)
#   - mixtral-8x7b-32768        (Mixtral 8x7B)
GROQ_MODEL=openai/gpt-oss-120b

# Vector Store Settings (Local ONNX Embeddings - No API Key Needed)
CHROMA_COLLECTION=enterprise-policy-assistant
TOP_K=5
CHUNK_SIZE=900
CHUNK_OVERLAP=150

# Guardrails & Security Controls
ENABLE_LLM_INJECTION_CHECK=true
ENABLE_GROUNDEDNESS_CHECK=true
GROUNDEDNESS_THRESHOLD=0.75
```

### 4. Build Policy Index
You can build or re-index the policy documents directly from the Streamlit UI or via command line:

```bash
python -c "from src.vector_store import rebuild_vector_store; rebuild_vector_store()"
```

### 5. Launch the Application
Start the Streamlit web dashboard:

```bash
streamlit run app.py
```

The app will be available in your browser at `http://localhost:8501`.

---

## 🐳 Docker Deployment

You can run Sentinel using either **Docker Compose** (recommended) or the **Docker CLI**.

### Option A: Using Docker Compose (Recommended)
Make sure your `GROQ_API_KEY` is configured in `.env`, then run:

```bash
# Build and start in background
docker compose up -d --build

# View container logs
docker compose logs -f

# Stop the container
docker compose down
```

### Option B: Using Docker CLI

```bash
# 1. Build the Docker image
docker build -t sentinel-policy-ai .

# 2. Run the container with your .env configuration
docker run -d \
  --name sentinel-policy-ai \
  -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/data/policies:/app/data/policies:ro \
  sentinel-policy-ai
```

Access the application in your browser at **`http://localhost:8501`**.

---

## 🧪 Running Automated Tests

Run the test suite with `pytest`:

```bash
python -m pytest tests/test_model_config.py -v
```

All 11 unit tests validate:
- Model alias resolution (`gpt-120` ➔ `openai/gpt-oss-120b`)
- `ChatGroq` structured output bindings
- Local offline embeddings query generation
- Deterministic prompt injection defense
- PII redaction
- Evidence formatting & citation synthesis
- Pipeline exception handling

---

## 🛡️ Security Controls & Guardrails

| Guardrail | Mechanism | Action |
| :--- | :--- | :--- |
| **Prompt Injection** | Deterministic Regex + Structured LLM Classification | Blocks malicious requests before retrieval |
| **PII Redaction** | Regex patterns for Email, PAN, Aadhaar, Phone, Credit Cards | Redacts matching entities to `[REDACTED_<TYPE>]` |
| **Groundedness Judge** | Structured LLM assessment evaluating claims against context | Flags or softens ungrounded hallucinations |
| **Context Isolation** | RAG prompt rules treating policy context as untrusted data | Prevents prompt injection via document text |

---

## 📄 License

This project is licensed under the Apache 2.0 License.
