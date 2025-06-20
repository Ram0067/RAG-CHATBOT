import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import PyPDF2

load_dotenv()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        pdf = PyPDF2.PdfReader(f)
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_and_store(doc_path):
    text = read_pdf(doc_path)
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            metadatas=[{"source": doc_path}],
            ids=[f"{os.path.basename(doc_path)}_{i}"]
        )
    print(f"Stored {len(chunks)} chunks from {doc_path}")

if __name__ == "__main__":
    embed_and_store("data/sample-documents/your_file.pdf")
    chroma_client.persist()
