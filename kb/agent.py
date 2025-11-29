import os
import requests
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from openai import OpenAI

# Use OpenRouter with OpenAI-Compatible Client
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def embed_text(texts):
    """Generate embeddings using OpenRouter via OpenAI-compatible API."""
    res = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in res.data]

class KBAgent:
    def __init__(self, chroma_dir="./chromadb_data"):
        self.chroma_dir = chroma_dir
        self.vectordb = Chroma(
            persist_directory=self.chroma_dir,
            embedding_function=embed_text
        )

    def answer(self, query):
        # retrieve
        docs = self.vectordb.similarity_search(query, k=4)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
You are a KnowledgeBase assistant. Use ONLY the context to answer.

Context:
{context}

Question:
{query}
"""

        # call OpenRouter LLM
        result = client.chat.completions.create(
            model="openai/gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return result.choices[0].message.content.strip()

def get_agent():
    return KBAgent()
