import os
import docx2txt
from PyPDF2 import PdfReader
import pdfplumber

def parse_resume(file_path: str) -> dict:
    
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    text = ""
    if ext == ".pdf":
        try:
            reader = pdfplumber.open(file_path)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            raise ValueError(f"Error parsing PDF: {e}")

    elif ext == ".docx":
        try:
            text = docx2txt.process(file_path)
        except Exception as e:
            raise ValueError(f"Error parsing DOCX: {e}")

    else:
        raise ValueError("Unsupported file type. Please use .pdf or .docx")

    # Basic cleaning
    text = text.replace("\xa0", " ").strip()

    return {
        "file_name": os.path.basename(file_path),
        "file_type": ext,
        "text": text
    }


# test

if __name__ == "__main__":
    """
    resume_path = "Your_Resume_File.pdf"  # Replace with your resume file path
    parsed = parse_resume(resume_path)
    print(parsed["file_name"], parsed["file_type"])
    print(parsed["text"])  # show preview
    """

