# Resume Parser

A Python-based parser that extracts structured information (name, contact, skills, education, experience, projects, certifications) from resumes in DOCX and PDF format.  
Uses PyMuPDF for PDF text, with Tesseract OCR fallback for scanned resumes.  
Normalizes dates, splits sections, and outputs both JSONL and CSV for analysis.  

## Features
- Handles both native PDFs and scanned PDFs (via OCR).  
- Extracts:  
  - Name  
  - Contact (emails, phones, links, location hints)  
  - Summary / Profile  
  - Skills (with smarter splitting)  
  - Education (degree, institution, dates, GPA/notes)  
  - Experience (title, company, location, bullets, normalized dates)  
  - Projects and Certifications  
- Outputs structured JSONL and quick CSV summaries.  

## Installation

### 1. Clone this repository
git clone https://github.com/yourusername/resume-parser.git
cd resume-parser

### 2. Install Python dependencies
pip install -r requirements.txt

Minimal requirements:
pymupdf
pdf2image
pytesseract
pillow
python-docx
docx2txt
dateparser

### 3. Install external tools
- Tesseract OCR → https://github.com/tesseract-ocr/tesseract  
  Windows: Install, or place the Tesseract-OCR folder inside your project.  
- Poppler for Windows (needed for pdf2image to rasterize PDFs):  
  Download from http://blog.alivate.com.au/poppler-windows/  
  Extract and add bin/ folder to PATH or set POPPLER_PATH in resume_parser.py.  

## Usage

### Parse a folder of resumes
python resume_parser.py ./Resumes -o parsed.jsonl --csv summary.csv

### Parse a single resume
python resume_parser.py "Bishop Resume 2024.docx"

### Options
- -o / --out → Output JSONL file (default: parsed_resumes.jsonl)  
- --csv → Output CSV summary file (default: parsed_summary.csv)  
- --ocr-lang → Tesseract language code (default: eng)  
- --pdf-density-threshold → Minimum chars/page to trust PDF text instead of OCR (default: 500)  

## Output Files
- JSONL → One JSON object per resume with full structured data.  
- CSV → Quick summary (name, emails, phones, links, top_skill, first_company, first_title).  

## Example
python resume_parser.py ./Resumes -o parsed.jsonl --csv summary.csv

Output preview (summary.csv):

| source_path      | name         | emails            | top_skill | first_company | first_title |
|------------------|--------------|-------------------|-----------|---------------|-------------|
| Bishop2024.docx  | Marc Bishop  | bishop@email.com  | Python    | OWT Global    | Engineer    |

## Troubleshooting
- OCR not working?  
  Check that tesseract.exe exists and is pointed correctly inside resume_parser.py.  
- PDF not parsed?  
  Install Poppler and set the path in the script.  
- Weird text artifacts (for example ligatures, dashes)?  
  OCRed resumes may need cleanup. Adjust regex or pre-process images.  

## For Developers
- Code is modular with comments:  
  - extract_* → text readers  
  - parse_* → structured field parsers  
  - split_sections → section detection  
- Add new section headers in SECTION_HEADERS (for example RESEARCH EXPERIENCE).  
- Extend KNOWN_ORGS to bias company/title detection.  
