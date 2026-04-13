import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", "./data/CONTRATO_SLA.txt")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./data/chroma_db")

def load_and_index_data():
    print(f"Cargando documento desde: {DOCUMENT_PATH}")
    loader = TextLoader(DOCUMENT_PATH, encoding="utf-8")
    documents = loader.load()

    # Estrategia de particionado: Se utiliza solapamiento (overlap) para asegurar
    # que no se pierda el contexto semántico en los límites de cada chunk.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150, 
        chunk_overlap=20,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Documento dividido en {len(chunks)} chunks.")

    # Selección de modelo de embeddings optimizado para inferencia en CPU.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Ingesta y persistencia de la base de datos vectorial para evitar 
    # recomputar los embeddings durante el ciclo de vida de la API.
    print("Creando base de datos vectorial ChromaDB...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    print(f"Base de datos guardada exitosamente en {CHROMA_DB_DIR}")

if __name__ == "__main__":
    load_and_index_data()