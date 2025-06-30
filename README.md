# 🔮 Google Gemini LLM App

## 👤 Author

- **Daniel Egbo** – [@Danselem](https://github.com/Danselem)

---
A demonstration project for building LLM applications using Google's **Gemini models**, with **LangChain**, **ChromaDB**, and **Retrieval-Augmented Generation (RAG)**. This app shows how to load documents, embed them, store in a vector DB, and answer queries based on context.

**Update**: Tracing has been integrated into the project with Arize Open Inference Telemetry to log traces and spans.



---

## 📦 Features

- ✅ Integration with **Google Gemini** (e.g., `gemini-1.5-flash`)
- 📄 PDF ingestion and preprocessing
- ✂️ Intelligent document chunking with LangChain
- 🧠 Vector embeddings using Gemini
- 🗃️ ChromaDB as the vector store
- 🔎 Semantic search for relevant document chunks
- 💬 Question-answering using a RAG pipeline
- 📈 **Tracing and observability** with [Arize Open Inference](https://github.com/Arize-ai/openinference) for end-to-end span-level logging. **Phoenix Open Inference has been persisted with PostgreSQL and has been containerized with Docker.** See the [docker-compose](docker-compose.yaml) file.
- 🧪 Includes example usage in the examples directory

**Note:** Sentence [transformer embeddings](https://huggingface.co/sentence-transformers) has been integrated into the project, see the [embedding module](src/embeddings/sentence_embedding.py). 

---

## 📁 Project Structure

```text
gemini_llm_app/
├── config
│   └── logging_config.yaml
├── data
│   ├── embeddings
│   ├── image
│   │   └── outputs
│   ├── outputs
│   │   ├── safety_output.txt
│   │   └── think_output.txt
│   ├── pdfs
│   │   └── p554.pdf
│   └── prompts
├── Dockerfile
├── examples
│   ├── __init__.py
│   ├── funccall.py
│   ├── ima_gen.py
│   ├── psumm.py
│   ├── psumm2.py
│   ├── quick.py
│   ├── rag
│   │   └── app.py
│   ├── safety.py
│   ├── search.py
│   ├── struct.py
│   └── think.py
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
├── requirements.txt
├── src
│   ├── embeddings
│   │   └── gemini_embedding.py
│   ├── handlers
│   │   ├── __init__.py
│   │   │   └── error_handler.cpython-311.pyc
│   │   └── error_handler.py
│   ├── llm
│   │   ├── gemini_client.py
│   │   ├── lang_gemini.py
│   │   └── utils.py
│   ├── prompt_engineering
│   │   ├── __init__.py
│   │   └── templates.py
│   └── utils
        ├── doc_loader.py
│       ├── doc_split.py
│       ├── download_file.py
│       ├── logger.py
│       └── rate_limiter.py
├── storage
│   └── chroma
│       └── chroma.sqlite3
└── uv.lock
```
---
## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Danselem/gemini_llm_app.git
cd gemini_llm_app
```
### 2. Create and Activate a Virtual Environment
This project uses the [`uv`](https://docs.astral.sh/uv/) project manager and [`make`](https://www.gnu.org/software/make/) tools.
```bash
uv venv --python 3.11
source .venv/bin/activate
```
### 3. Install Dependencies
```bash
uv pip install --all-extras --requirement backend/pyproject.toml
```
or
```bash
make install
```
---
## 🔐 Environment Variables
Create a `.env` file at the project root and fill the environment variables with `make env`.
```env
GOOGLE_API_KEY=your_gemini_api_key
MULTIMODAL_MODEL=gemini-1.5-flash
```
---

## 📈 Start the Telemetry Server
This project uses the `Arize` Open Inference for telemetry and logging of traces and spans. To start the telemetry server, run the command below.
```bash
make start-phoenix
```
Note, ensure your `Docker` software is running before running the above command.

---
## ▶️ Run the App
There are multiple examples in the `examples` directory for you to get started with, e.g.:

```bash
python -m examples.psumm
```
or
```bash
make psumm
```

To run the `RAG` app, use the command below:
```bash
make app
```
The example code summarises a pdf file. There are multiple examples in the `examples` directory. You can also check out the `Makefile` to see other examples.

---

## ⚙️ How It Works
### 1. Download & Parse PDF
The PDF is downloaded from a URL if it's not already in `data/pdfs`.

### 2. Text Splitting
The PDF is split into overlapping chunks using `RecursiveCharacterTextSplitter` from LangChain.

### 3. Embeddings
Each chunk is converted into a vector using Google Gemini Embedding or Sentence Transformer Embedding (via LangChain wrapper).

### 4. Vector Store
The chunks and their embeddings are stored in a local ChromaDB collection.

### 5. Semantic Search
When a user asks a question, the app retrieves relevant chunks using vector similarity search.

### 6. RAG Response
The relevant chunks are passed to Gemini to answer the question contextually.

---
## 📚 Tech Stack
* LangChain

* ChromaDB

* Phoenix

* Open Inference

* Google Gemini API

* Poppler

* Python 3.11+

---

## 📄 License
MIT License. See the [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgements
* Google for releasing the Gemini family of models

* LangChain community for open-source tools

* ChromaDB team for fast and easy vector storage

---

## 💡 Ideas for Extension
* 🔧 Add a simple web UI with Gradio or Streamlit for the RAG application.

* 📝 Ingest multiple documents and support multi-source QA.

* 🧠 Add caching to avoid re-embedding on re-runs.

* 📊 Integrate telemetry/tracing and observability with `Open Inference and Phoenix.