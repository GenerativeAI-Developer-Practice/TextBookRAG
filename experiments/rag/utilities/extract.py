import fitz  # PyMuPDF
from PIL import Image
import os
import io
import numpy as np

def is_blank_image(pil_img, threshold=10):
    """Returns True if the image is almost blank (white or very low variance)."""
    gray_img = pil_img.convert('L')
    np_img = np.array(gray_img)
    stddev = np.std(np_img)
    return stddev < threshold

def extract_text_and_images(pdf_path, output_dir):
    """Extract text and images
    """
    os.makedirs(output_dir, exist_ok=True)
    text_dir = os.path.join(output_dir, "text")
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # Save text
        text = page.get_text()
        with open(os.path.join(text_dir, f"page_{page_num+1}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

        # Save non-blank images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_ext = base_image["ext"]
            image = Image.open(io.BytesIO(image_bytes))

            if not is_blank_image(image):
                image.save(os.path.join(image_dir, f"page_{page_num+1}_img_{img_index+1}.{img_ext}"))
    
    print(f"✅ Extraction complete. Text in '{text_dir}', Images in '{image_dir}'")

