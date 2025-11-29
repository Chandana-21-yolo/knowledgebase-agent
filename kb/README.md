# KnowledgeBase Agent — OpenRouter Edition

This repository contains a Streamlit KnowledgeBase Agent that:
- Ingests PDFs / TXT files into a Chroma vector store
- Uses OpenAI embeddings (via the OpenAI API key)
- Uses **OpenRouter** for LLM completions (set `OPENROUTER_API_KEY`)
- Streamlit UI for asking questions and uploading documents

## Files
- `streamlit_app.py` — Streamlit UI (upload, ingest, query)
- `ingest.py` — Ingest documents from `./docs` into Chroma
- `agent.py` — Retrieval + OpenRouter prompt wrapper
- `requirements.txt` — Python deps
- `example_config.env` — Example env vars
- `architecture.txt` — Architecture notes & deployment tips

## Setup (local)
1. Create `.env` or set secrets in Streamlit Cloud:
   ```
   OPENROUTER_API_KEY="sk-or-..."
   OPENAI_API_KEY="sk-or-..."   # use same key if you only have OpenRouter key
   CHROMA_PERSIST_DIR="./chromadb_data"
   ```
2. Install:
   ```
   pip install -r requirements.txt
   ```
3. Add docs:
   ```
   mkdir docs
   # put your PDF/TXT files into ./docs
   python ingest.py
   ```
4. Run Streamlit:
   ```
   streamlit run streamlit_app.py
   ```

## Deploying to Streamlit Cloud
- Push this repo to GitHub.
- In Streamlit Cloud, set **Main file** to `streamlit_app.py`.
- Add Secrets (TOML):
```
OPENROUTER_API_KEY="sk-or-..."
OPENAI_API_KEY="sk-or-..."
```
- Deploy.

## Notes
- Embeddings use `OpenAIEmbeddings` from LangChain and require `OPENAI_API_KEY`. If you only have an OpenRouter key, set both to the same value.
- For production, replace Chroma local persistence with Pinecone/Weaviate.

