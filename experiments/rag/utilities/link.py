import json

def link_text_and_images(text_json_file, image_json_file):
    """
    Loads text chunk and image embedding data from JSON files,
    groups them by page_number, and returns a combined dictionary.
    """
    with open(text_json_file, "r", encoding="utf-8") as f:
        text_data = json.load(f)
    with open(image_json_file, "r", encoding="utf-8") as f:
        image_data = json.load(f)
    
    # Group text chunks by page number
    text_by_page = {}
    for item in text_data:
        page = item.get("page_number")
        if page is not None:
            text_by_page.setdefault(page, []).append(item)
    
    # Group images by page number
    images_by_page = {}
    for item in image_data:
        page = item.get("page_number")
        if page is not None:
            images_by_page.setdefault(page, []).append(item)
    
    # Combine the groups: For every page, include both text chunks and images.
    linked_data = {}
    all_pages = set(text_by_page.keys()).union(images_by_page.keys())
    for page in all_pages:
        linked_data[page] = {
            "text_chunks": text_by_page.get(page, []),
            "images": images_by_page.get(page, [])
        }
    
    return linked_data


