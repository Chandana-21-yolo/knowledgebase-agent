import os
import requests
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# OpenRouter chat completions endpoint
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def ask_openrouter(prompt, api_key, model="openai/gpt-4.1-mini", temperature=0.0):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Expecting structure similar to OpenAI-compatible response
    if "choices" in data and len(data["choices"])>0:
        # Some OpenRouter responses return message object under choices[0]["message"]["content"]
        choice = data["choices"][0]
        if isinstance(choice.get('message'), dict):
            return choice['message'].get('content', '').strip()
        # fallback
        return choice.get('text', '').strip() or str(data)
    return str(data)

class KBAgent:
    def __init__(self, chroma_dir='./chromadb_data', openai_api_key=None, openrouter_key=None, model='openai/gpt-4.1-mini'):
        self.chroma_dir = chroma_dir
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.openrouter_key = openrouter_key or os.getenv('OPENROUTER_API_KEY')
        self.model = model
        # Initialize embeddings + vectordb
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.vectordb = Chroma(persist_directory=self.chroma_dir, embedding_function=self.embeddings)

    def answer(self, query, k=4):
        # Retrieve top-k documents
        docs = self.vectordb.similarity_search(query, k=k)
        context = "\n\n---\n\n".join([d.page_content for d in docs])
        prompt = f"""You are a helpful KnowledgeBase assistant. Use ONLY the context below to answer the question. If the answer is not in the context, say you don't know.

CONTEXT:
{context}

QUESTION:
{query}

Provide a concise answer and include a short 'source' listing which documents (by index) were used.
"""
        return ask_openrouter(prompt, api_key=self.openrouter_key, model=self.model)

def get_agent(chroma_dir='./chromadb_data'):
    return KBAgent(chroma_dir=chroma_dir)
