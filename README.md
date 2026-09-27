# DocuMind RAG

A Streamlit application that allows users to upload PDF documents and ask questions about their contents using retrieval-augmented generation.

## Features

- Upload PDF documents
- Extract and split PDF text
- Store document embeddings in Chroma
- Ask questions through a Streamlit chat interface
- Display referenced PDF pages

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file in the project directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Run the application:

```powershell
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Security

Do not commit `.env` or expose your OpenAI API key.
