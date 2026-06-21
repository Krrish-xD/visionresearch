import os
import chromadb
from .memory_utils import clear_memory

def run_rag_retrieval(z3_facts, output_dir):
    """
    Stage 4: Local RAG
    Queries hardcoded universal logic based on Z3 facts.
    """
    print("[Stage 4] Initializing ChromaDB for local RAG...")
    
    # Persistent storage in data/db
    db_path = os.path.abspath(os.path.join(output_dir, "..", "db"))
    os.makedirs(db_path, exist_ok=True)
    
    client = chromadb.PersistentClient(path=db_path)
    
    # Create or get collection
    collection = client.get_or_create_collection(name="universal_logic")
    
    # Hardcode 3 mock chunks
    documents = [
        "When Object A's major axis vector aligns with the vector to Object B, A is considered 'facing' B.",
        "If Object A and B intersect in 2D space, but A's depth value is lower than B's, A is physically occluding B.",
        "The relative Euclidean distance between 3D centroids determines spatial proximity in the metric space."
    ]
    ids = ["doc_1", "doc_2", "doc_3"]
    
    if collection.count() == 0:
        collection.add(
            documents=documents,
            ids=ids
        )
        
    print("[Stage 4] Querying DB with Z3 facts...")
    
    if not z3_facts or "No spatial relationships proven" in z3_facts:
        clear_memory()
        return "No relevant spatial context found."
        
    results = collection.query(
        query_texts=[z3_facts],
        n_results=2
    )
    
    retrieved_docs = results['documents'][0] if results['documents'] else []
    
    context_string = "\n".join([f"- {doc}" for doc in retrieved_docs])
    
    print("[Stage 4] Context retrieved.")
    clear_memory()
    
    return context_string
