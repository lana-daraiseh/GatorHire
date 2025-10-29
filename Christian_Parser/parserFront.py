import ollama
import re
import json
import csv
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tempParser import parse_resume
import io
import re
from datetime import datetime
import pandas as pd
import streamlit as st
from pypdf import PdfReader
import pdfplumber
import json
import tempfile
import time

import fitz


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

def main():
    st.set_page_config(page_title="AI Resume Parser", layout="wide")
    st.title("AI Resume Parser")
    st.write("Upload one or more resumes. We’ll extract text and pull several fields for their analysis.")

    uploaded_files = st.file_uploader("Upload Resume Files", type=["pdf", "docx"], accept_multiple_files=True)

    if uploaded_files:
        results = []
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)

        for i, uploaded_file in enumerate(uploaded_files, start=1):
            # Save file temporarily (Windows-safe)
            suffix = os.path.splitext(uploaded_file.name)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
            try:
                tmp.write(uploaded_file.read())
                tmp.close()  # Important: close before re-opening in process_resume()

                # Process with your parser
                with st.spinner(f"Parsing {uploaded_file.name}..."):
                    result = process_resume(tmp.name)
                    results.append(result)

                progress_bar.progress(i / total_files)

            finally:
                # Safe cleanup — handle potential Windows file lock
                try:
                    os.remove(tmp.name)
                except PermissionError:
                    time.sleep(0.5)
                    try:
                        os.remove(tmp.name)
                    except Exception:
                        pass

            # Display structured data preview
            st.markdown(f"### {uploaded_file.name}")
            if "error" in result:
                st.error(f"Error parsing {uploaded_file.name}: {result['error']}")
            else:
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Contact Info:**")
                    st.write(f"**Name:** {result.get('name', '')}")
                    st.write(f"**Email:** {result.get('email', '')}")
                    st.write(f"**Phone:** {result.get('phone', '')}")

                    st.write("**Summary:**")
                    st.write(result.get("summary", "(none)"))

                    st.write("**Skills:**")
                    st.write(result.get("skills", "(none)"))

                with col2:
                    st.write("**Education:**")
                    try:
                        for edu in json.loads(result.get("education", "[]")):
                            st.write(f"- {edu.get('degree', '')}, {edu.get('school', '')} ({edu.get('year', '')})")
                    except Exception:
                        st.write(result.get("education", "(invalid JSON)"))

                    st.write("**Relevant Coursework:**")
                    st.write(result.get("relevant_classwork", "(none)"))

                # Expanders for Experience, Projects, Research
                with st.expander("Experience"):
                    try:
                        experiences = json.loads(result.get("experience", "[]"))
                        if experiences:
                            for exp in experiences:
                                st.markdown(f"**{exp.get('title', '')}** — {exp.get('company', '')}")
                                st.write(exp.get("dates", ""))
                                for b in exp.get("bullets", []):
                                    st.markdown(f"- {b}")
                        else:
                            st.write("(No experience found)")
                    except Exception:
                        st.write(result.get("experience", "(invalid JSON)"))

                with st.expander("Projects"):
                    try:
                        projects = json.loads(result.get("projects", "[]"))
                        if projects:
                            for proj in projects:
                                st.markdown(f"**{proj.get('name', '')}**")
                                st.write(proj.get("description", ""))
                        else:
                            st.write("(No projects found)")
                    except Exception:
                        st.write(result.get("projects", "(invalid JSON)"))

                with st.expander("Research"):
                    try:
                        research = json.loads(result.get("research", "[]"))
                        if research:
                            for r in research:
                                st.markdown(f"**{r.get('title', '')}**")
                                st.write(r.get("description", ""))
                        else:
                            st.write("(No research found)")
                    except Exception:
                        st.write(result.get("research", "(invalid JSON)"))

        # Final CSV export
        if results:
            df = pd.DataFrame(results)
            st.subheader("Parsed Summary Table")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results as CSV",
                data=csv,
                file_name="parsed_resumes.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()