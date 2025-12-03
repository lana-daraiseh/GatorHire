import ollama
import re
import json
import csv
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from textExtractor import parse_resume


# ---------- Setup Logging ----------
def setup_logger(log_file="parser_log.log"):
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )
    logging.getLogger().addHandler(logging.StreamHandler())  # print to console
    logging.info("Resume Parsing Session Started")


# ---------- Ollama Extraction ----------
def extract_resume_with_ollama(resume_text: str, model: str = "gemma3:12b", max_retries: int = 3) -> dict:
    """
    Extract structured resume info using Ollama with retries and JSON cleanup.
    """

    prompt_template = f"""
    You are a resume parser. Extract structured information from the text below. 
    Return JSON strictly in this schema, do not grab information from outside its designated section (no markdown, no explanation, only valid JSON):

    {{
      "contact": {{"name":"", "email":"", "phone":"", "links":[]}},
      "summary": "",
      "experience": [
        {{"title":"", "company":"", "dates":"", "bullets":[""]}}
      ],
      "projects": [{{"name":"", "description":""}}],
      "research": [{{"title":"", "description":""}}],
      "relevant_classwork": [""],
      "education": [{{"school":"", "degree":"", "GPA":"", "year":""}}],
      "skills": []
    }}

    Resume Text:
    {resume_text}
    """

    for attempt in range(1, max_retries + 1):
        try:
            response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt_template}])
            raw_output = response.message.content.strip()

            # Remove markdown if needed
            raw_output = re.sub(r"^```(?:json)?|```$", "", raw_output, flags=re.MULTILINE).strip()

            # Extract only JSON portion
            if "{" in raw_output:
                raw_output = raw_output[raw_output.find("{"):]

            parsed = json.loads(raw_output)
            return parsed  # Success

        except Exception as e:
            logging.warning(f"Retry {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                logging.error(f"Failed to parse with Ollama after {max_retries} retries.")
                return {"raw": raw_output if 'raw_output' in locals() else str(e)}

    return {"error": "Unknown failure"}


# ---------- Single Resume Processing ----------
def process_resume(filepath: str) -> dict:
    """
    Process a single resume and extract structured data.
    """
    filename = os.path.basename(filepath)
    logging.info(f"Processing file: {filename}")

    try:
        parsed = parse_resume(filepath)
        ollama_structured = extract_resume_with_ollama(parsed["text"])

        result = {
            "filename": filename,
            "name": ollama_structured.get("contact", {}).get("name", ""),
            "email": ollama_structured.get("contact", {}).get("email", ""),
            "phone": ollama_structured.get("contact", {}).get("phone", ""),
            "summary": ollama_structured.get("summary", ""),
            "skills": ", ".join(ollama_structured.get("skills", [])),
            "education": json.dumps(ollama_structured.get("education", []), ensure_ascii=False),
            "experience": json.dumps(ollama_structured.get("experience", []), ensure_ascii=False),
            "projects": json.dumps(ollama_structured.get("projects", []), ensure_ascii=False),
            "research": json.dumps(ollama_structured.get("research", []), ensure_ascii=False),
            "relevant_classwork": ", ".join(ollama_structured.get("relevant_classwork", [])),
        }

        logging.info(f"Successfully parsed: {filename}")
        return result

    except Exception as e:
        logging.error(f"Error processing {filename}: {e}")
        return {"filename": filename, "error": str(e)}


# ---------- Batch Processor ----------
def batch_parse_resumes(folder_path: str, output_csv: str, log_file: str = "parser_log.log", limit: int = 20, max_workers: int = 5):
    """
    Parse multiple resumes concurrently, with logging and CSV export.
    """
    setup_logger(log_file)

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.pdf', '.docx'))]
    files = files[:limit]
    total_files = len(files)

    logging.info(f"Found {total_files} resumes in folder '{folder_path}'. Starting batch parse...")

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_resume, os.path.join(folder_path, f)): f for f in files}

        for future in tqdm(as_completed(future_to_file), total=total_files, desc="Parsing Resumes", ncols=100):
            result = future.result()
            results.append(result)

    # ---------- Write to CSV ----------
    fieldnames = [
        "filename", "name", "email", "phone", "summary",
        "skills", "education", "experience", "projects",
        "research", "relevant_classwork"
    ]

    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logging.info(f"Parsing complete! {len(results)} resumes processed. Output saved to '{output_csv}'.")
    print(f"\nDone! Parsed {len(results)} resumes. Saved to '{output_csv}'. Log: '{log_file}'")


# ---------- Run ----------
if __name__ == "__main__":
    """
    BATCH PROCESSING EXAMPLE USAGE:
    folder_path = "resumes"
    output_csv = "parsed_resumes.csv"
    log_file = "parser_log.log"

    batch_parse_resumes(folder_path, output_csv, log_file, limit=20, max_workers=5) # Adjust limit and workers as needed
    
    SINGLE FILE PROCESSING EXAMPLE USAGE:
    ollama_structured = process_resume("Your_Resume_File.pdf")
    print(json.dumps(ollama_structured, indent=2, ensure_ascii=False)) # Print the structured data
    """
