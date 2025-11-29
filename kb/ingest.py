import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def embed_text(texts):
    res = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in res.data]

def ingest():
    folder = "./docs"
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".pdf") or f.endswith(".txt")
    ]

    if not files:
        print("No documents found.")
        return

    docs = []
    for f in files:
        if f.endswith(".pdf"):
            loader = PyPDFLoader(f)
        else:
            loader = TextLoader(f)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        chunks,
        embedding_function=embed_text,
        persist_directory="./chromadb_data"
    )

    vectordb.persist()
    print("Ingestion completed.")

if __name__ == "__main__":
    ingest()
