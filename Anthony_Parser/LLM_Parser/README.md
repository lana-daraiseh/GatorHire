# Experimental LLM Resume Parser  
- This folder contains a local-only parser that extracts strcutured JSON data from '.pdf', '.docx', or '.txt' resumes.  
- It does this using a local Ollama model (default is 'llama3.1:8b').  


# Requirements  
- Ollama server must be running locally (command: 'ollama serve') and a model pulled (command: 'ollama pull llama3.1:8b')  
- (Optional) [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) for scanned/image pdfs  


# Usage  
## Command Line  
Use directly:   
- python LLM_Parser.py --in resume.pdf  

Enable OCR for scanned/image pdfs:  
- python LLM_Parser.py --in scanned_resume.pdf --ocr

Choose specific model:  
- python LLM_Parser.py --in resume.pdf --model llama3.1:8b

## Python

- from LLM_Parser import extract_resume_info_simple  
- data = extract_resume_info_simple("resume.pdf")  
- print(data)  