import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from data_loader import load_markdown_files

load_dotenv()

# CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
# EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
# DEFAULT_LINK = os.getenv("DEFAULT_COMPANY_LINK", "https://www.formaculture.com/about")

CHROMA_DIR="./data/chroma_db"

# text=load_markdown_files()
# print(text)

def create_or_update_vectorstore(persist_directory: str = CHROMA_DIR, force_reingest: bool = False):
    embeddings = OpenAIEmbeddings()

    # If DB already exists on disk, just load it
    if not force_reingest and os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("Loading existing Chroma DB from disk.")
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Otherwise, build from scratch
    texts = load_markdown_files()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(texts)
    print("Creating new Chroma DB...")
    db = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=persist_directory)
    db.persist()
    return db

if __name__ == "__main__":
    db = create_or_update_vectorstore()
    print("Ingestion complete. Chroma persisted at:", CHROMA_DIR)