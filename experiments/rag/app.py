import streamlit as st
import json
import numpy as np
import faiss
import os
from PIL import Image
from sentence_transformers import SentenceTransformer


def aggregate_context(results, image_data):
    context_text = ""
    context_images = ""
    for result in results:
        page = result.get("page_number")
        context_text += f"Page {page} - {result.get('text_chunk')}\n\n"
        linked_images = [img for img in image_data if img.get("page_number") == page]
        if linked_images:
            # For now, use image filenames as reference; later, replace with generated captions.
            context_images += f"Page {page} Images: " + ", ".join([img.get("image_file") for img in linked_images]) + "\n\n"
    return context_text.strip(), context_images.strip()


def construct_prompt(query, text_context, image_context):
    prompt = f"""
You are an expert tutor. Use the following textbook context to answer the query thoroughly.

Text Context:
{text_context}

Image Context:
{image_context}

Question: {query}

Answer:
"""
    return prompt




# -------- Functions for Loading and Building FAISS Indices --------

def load_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

def build_faiss_index(data, embedding_field="embedding"):
    """
    Build a FAISS index from a list of data items that include the given embedding_field.
    Returns both the FAISS index and the numpy array of embeddings.
    """
    embeddings = [item[embedding_field] for item in data]
    embeddings_np = np.array(embeddings).astype("float32")
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    st.write(f"✅ FAISS Index built: {index.ntotal} items, Dimension {dimension}")
    return index

def search_text_index(query, index, data, model, top_k=5):
    """
    Embed the query using the model and search the text FAISS index.
    Returns a list of the top_k matching data items.
    """
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i in indices[0]:
        if i < len(data):
            results.append(data[i])
    return results

def link_text_with_images(text_item, image_data):
    """
    Given a text item (which contains a page_number), 
    return all image items that share the same page_number.
    """
    page = text_item.get("page_number")
    linked_images = [img for img in image_data if img.get("page_number") == page]
    return linked_images


from transformers import pipeline

# Initialize the text generation pipeline (adjust max_length as needed)
generator = pipeline("text-generation", model="gpt2", max_length=500)

def generate_answer(prompt):
    response = generator(prompt, max_length=500, do_sample=True, temperature=0.7)
    # Extract the generated text from the response.
    return response[0]['generated_text']


# -------- Initialization --------

st.title("RAG Textbook Chatbot")
st.write("Ask a question to search the textbook content and view the related images.")

# Load the SentenceTransformer model (for text queries)
@st.cache_resource  # streamlit caching for models
def load_text_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model_text = load_text_model()

# Load the text embeddings data and build the index
text_json_path = "text_chunks_with_embeddings.json"
text_data = load_json(text_json_path)
text_index = build_faiss_index(text_data, "embedding")

# Load the image embeddings data
image_json_path = "image_embeddings_with_metadata.json"
image_data = load_json(image_json_path)

# Define the folder path for images as extracted by Step 1
IMAGE_FOLDER = "extracted/images"

# -------- Streamlit Query Interface --------

st.title("RAG Textbook Chatbot with Generative Answers")
st.write("Ask a question to search the textbook content and generate a detailed answer.")

# Query input (this part is similar to before)
query = st.text_input("Enter your query:", value="Explain Newton's first law")
if st.button("Search"):
    if query:
        st.write("Searching for relevant content...")
        results = search_text_index(query, text_index, text_data, model_text, top_k=5)

        if results:
            st.subheader("Retrieved Textual Results:")
            for idx, result in enumerate(results):
                st.markdown(f"**Result {idx+1}** (Page {result.get('page_number', 'Unknown')}, Chunk {result.get('chunk_id')})")
                st.write(result.get("text_chunk"))
                linked_images = link_text_with_images(result, image_data)
                if linked_images:
                    st.markdown("**Related Images:**")
                    cols = st.columns(len(linked_images))
                    for i, img_meta in enumerate(linked_images):
                        img_file = os.path.join(IMAGE_FOLDER, img_meta.get("image_file"))
                        if os.path.exists(img_file):
                            cols[i].image(Image.open(img_file), caption=img_meta.get("image_file"), use_column_width=True)
                        else:
                            cols[i].write(f"Image not found: {img_meta.get('image_file')}")
                st.markdown("---")

            # Aggregate context from results
            text_context, image_context = aggregate_context(results, image_data)
            
            # Construct the prompt for the generative model
            prompt = construct_prompt(query, text_context, image_context)
            st.text_area("Constructed Prompt:", prompt, height=300)

            # Button to generate answer
            if st.button("Generate Answer"):
                # Initialize the generator (you might cache this as well)
                generator = pipeline("text-generation", model="gpt2", max_length=500)
                answer = generate_answer(prompt)
                st.subheader("Generated Answer:")
                st.write(answer)
        else:
            st.write("No results found for your query.")
    else:
        st.write("Please enter a query.")

