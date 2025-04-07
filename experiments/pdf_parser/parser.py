import fitz  # PyMuPDF

def parse_pdf_page_by_page(pdf_path, output_dir="output"):
    """
    Parses a PDF file page by page, extracting text and images.

    Args:
        pdf_path (str): Path to the PDF file.
        output_dir (str): Directory to save extracted images.
    """

    try:
        pdf_document = fitz.open(pdf_path)

        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]

            # Extract Text
            text = page.get_text("text")  # or "blocks", "words", etc.
            print(f"--- Page {page_number + 1} Text ---")
            print(text)

            # Extract Images
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                image_filename = f"{output_dir}/page_{page_number + 1}_img_{img_index + 1}.{image_ext}"
                with open(image_filename, "wb") as image_file:
                    image_file.write(image_bytes)
                print(f"Image saved: {image_filename}")

        pdf_document.close()

    except Exception as e:
        print(f"An error occurred: {e}")


pdf_file_path = "data/jesc109.pdf" # Replace with your PDF File path.

parse_pdf_page_by_page(pdf_file_path, "extracted_content") # You need to create the extracted_content folder.