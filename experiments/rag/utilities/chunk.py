import os
import re
import glob
import json

def chunk_text(text, max_length=500, overlap=50):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + max_length, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max_length - overlap
    return chunks

def extract_page_number(filename):
    """
    Extract the page number from a filename formatted as 'page_{page_number}.txt'
    """
    base = os.path.basename(filename)
    match = re.match(r"page_(\d+)", base)
    if match:
        return int(match.group(1))
    return None

def process_text_files_with_metadata(text_folder, model, output_file="text_chunks_with_embeddings.json"):
    data = []  # List to hold the chunks with metadata
    
    # Process all text files in the folder
    txt_files = glob.glob(os.path.join(text_folder, "*.txt"))
    for txt_file in txt_files:
        # Extract the page number from the file name
        page_number = extract_page_number(txt_file)
        
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Chunk the text from the file
        chunks = chunk_text(text, max_length=500, overlap=50)
        
        # Compute embeddings for each chunk
        embeddings = model.encode(chunks, show_progress_bar=True).tolist()
        
        # Add each chunk with its metadata to the data list
        for idx, chunk in enumerate(chunks):
            data.append({
                "source_file": os.path.basename(txt_file),
                "page_number": page_number,
                "chunk_id": idx,
                "text_chunk": chunk,
                "embedding": embeddings[idx]
            })
    
    # Save the output to a JSON file
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(data, out_f, indent=2)
    
    print(f"Processed {len(txt_files)} files. Data saved to '{output_file}'")
