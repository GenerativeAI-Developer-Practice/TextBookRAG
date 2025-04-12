import json
import numpy as np
import faiss

def build_faiss_index(json_file, embedding_field="embedding"):
    """
    Loads data from a JSON file where each item includes an embedding,
    builds a FAISS index, and returns both the index and the loaded data.

    Args:
      json_file (str): Path to the JSON file.
      embedding_field (str): The field key in the JSON items that contains the embedding.

    Returns:
      index (faiss.Index): The FAISS index.
      data (list): The JSON data as a list of dictionaries.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract embeddings from each item
    embeddings = [item[embedding_field] for item in data]
    embeddings_np = np.array(embeddings).astype("float32")
    
    # Determine the dimension of the embeddings
    dimension = embeddings_np.shape[1]
    
    # Create a flat (brute-force) L2 distance index
    index = faiss.IndexFlatL2(dimension)
    
    # Add all embeddings into the index
    index.add(embeddings_np)
    
    print(f"✅ Indexed {index.ntotal} text items with dimension {dimension}")
    return index, data

def build_faiss_index_for_images(json_file, embedding_field="embedding"):
    """
    Loads image data from a JSON file and builds a FAISS index for image embeddings.
    
    Args:
      json_file (str): Path to the JSON file containing image embeddings.
      embedding_field (str): The key that contains the embedding in each JSON object.
      
    Returns:
      index (faiss.Index): The FAISS index.
      data (list): The JSON data as a list of dictionaries.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    embeddings = [item[embedding_field] for item in data]
    embeddings_np = np.array(embeddings).astype("float32")
    dimension = embeddings_np.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    
    print(f"✅ Indexed {index.ntotal} image items with dimension {dimension}")
    return index, data

def search_text_index(query, index, metadata, model, top_k=5):
    """
    Searches the FAISS text index using a query string.
    
    Args:
      query (str): The user query.
      index (faiss.Index): The FAISS index for text embeddings.
      metadata (list): The list of metadata corresponding to the indexed items.
      model: The embedding model (SentenceTransformer) used to encode the query.
      top_k (int): The number of results to return.
      
    Returns:
      results (list): Top matching metadata items.
    """
    # Compute the query embedding
    query_embedding = model.encode([query]).astype("float32")
    
    # Search the FAISS index
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i in indices[0]:
        # Verify index validity and append corresponding metadata
        if i < len(metadata):
            results.append(metadata[i])
    return results





