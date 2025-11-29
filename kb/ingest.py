import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

def load_files(folder='./docs'):
    files = []
    for root,_,filenames in os.walk(folder):
        for f in filenames:
            if f.lower().endswith('.pdf') or f.lower().endswith('.txt'):
                files.append(os.path.join(root,f))
    return files

def ingest(docs_folder='./docs', chroma_dir='./chromadb_data'):
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY must be set (for embeddings).')
    paths = load_files(docs_folder)
    if not paths:
        print('No documents found in', docs_folder)
        return
    all_docs = []
    for p in tqdm(paths, desc='Loading files'):
        if p.lower().endswith('.pdf'):
            loader = PyPDFLoader(p)
        else:
            loader = TextLoader(p, encoding='utf8')
        all_docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)
    embed = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = Chroma.from_documents(chunks, embed, persist_directory=chroma_dir)
    vectordb.persist()
    print(f'Ingested {len(chunks)} chunks into {chroma_dir}')

if __name__ == '__main__':
    ingest()
