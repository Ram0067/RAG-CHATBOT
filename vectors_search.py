import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to persistent ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Make sure to access the collection after initializing the client
collection = chroma_client.get_or_create_collection(name="faq_docs")

# Function to perform vector search
def search_similar_chunks(query, top_k=3):
    embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return results

# Run test if executed directly
if __name__ == "__main__":
    user_query = input("Ask a question: ")
    results = search_similar_chunks(user_query)

    print("\nTop Relevant Chunks:\n")
    for i, doc in enumerate(results["documents"][0]):
        print(f"Result {i+1}:")
        print(doc)
        print(f"Similarity Score: {results['distances'][0][i]:.4f}")
        print("-" * 40)
