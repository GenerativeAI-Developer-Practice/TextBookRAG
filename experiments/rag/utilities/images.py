import os
import glob
import json
import re
from PIL import Image

def extract_page_and_index(filename):
    """
    Parses the filename to extract page number and image index.
    
    Expected filename format: page_{page_number}_img_{image_index}.{ext}
    
    Returns:
      (page_number, image_index) if available, otherwise (None, None).
    """
    base = os.path.basename(filename)
    # Use regex to extract page and image numbers
    match = re.match(r"page_(\d+)_img_(\d+)", base)
    if match:
        page_number = int(match.group(1))
        image_index = int(match.group(2))
        return page_number, image_index
    return None, None

def process_images_with_metadata(image_folder, model, output_file="image_embeddings_with_metadata.json"):
    """
    Processes images by:
      - Loading each image.
      - Computing its embedding using the provided model.
      - Extracting metadata (page number and image index) from the filename.
      - Saving the results in a JSON file.
    
    Args:
      image_folder (str): Folder containing the images.
      model: A SentenceTransformer model (e.g., CLIP-based) for image embeddings.
      output_file (str): Filename for saving the JSON output.
    """
    data = []
    # Adjust file extensions as needed
    image_files = glob.glob(os.path.join(image_folder, "*.*"))
    
    for image_file in image_files:
        try:
            # Open image and convert to RGB (required for models like CLIP)
            img = Image.open(image_file).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
            continue
        
        # Generate image embedding using the model (expects list input)
        embedding = model.encode([img], convert_to_tensor=False)[0].tolist()
        
        # Extract page and image index from filename (if available)
        page_number, image_index = extract_page_and_index(image_file)
        
        # Create the metadata dictionary
        meta = {
            "image_file": os.path.basename(image_file),
            "embedding": embedding,
            "page_number": page_number,
            "image_index": image_index
        }
        
        data.append(meta)
    
    # Save the enriched image metadata to a JSON file
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(data, out_f, indent=2)
    
    print(f"Processed {len(image_files)} images and saved metadata to '{output_file}'")


