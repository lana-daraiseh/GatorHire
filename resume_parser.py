import json
import os
import argparse
import pdfplumber

from regex import Regex
from pathlib import Path

class ResumeParser:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.text = self._extract_text()

    def _extract_text(self):
        text = []
        with pdfplumber.open(self.file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text).strip()      #.strip() ignores whitespace

    def extract_email(self):
        match = Regex.EMAIL.search(self.text)
        return match.group(0) if match else None

    def extract_phone(self):
        match = Regex.PHONE.search(self.text)
        return match.group(0) if match else None

    def extract_name(self):
        first_line = self.text.split("\n")[0].strip()                   # assume name is first line
        if len (first_line.split()) <= 4 and "@" not in first_line:
            return first_line
        else:
            return None

    def extract_skills(self):
        found_skills = []
        text_lower = self.text.lower()
        for skill in Regex.SKILLS:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        return found_skills

    def extract_education(self):
        lines = self.text.splitlines()
        education = [line for line in lines if "University" in line or "College" in line]
        return education

    def extract_experience(self):
        lines = self.text.splitlines()
        experience = [line for line in lines if Regex.YEAR.search(line)]
        return experience

    def parse(self):
        return {
            "Name": self.extract_name(),
            "Email": self.extract_email(),
            "Phone": self.extract_phone(),
            "Education": self.extract_education(),
            "Skills": self.extract_skills(),
            "Experience": self.extract_experience(),
        }


def save_json(data, pdf_path, out_dir="parsed_resumes"):
    os.makedirs(out_dir, exist_ok=True)
    out_name = Path(pdf_path).stem + ".json"
    out_path = Path(out_dir) / out_name
    with open(out_path, "w", encoding="utf-8") as f:    #encoding to handle dashes in between dates
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Parser")
    parser.add_argument("pdf", type=Path, nargs="*", default=["Resumes"], help="Paths to PDFs")
    args = parser.parse_args()

    pdf_inputs = args.pdf

    all_pdfs = []
    for p in pdf_inputs:
        p = Path(p)
        if p.is_dir():
            all_pdfs.extend(p.glob("*.pdf"))
        elif p.suffix.lower() == ".pdf":
            all_pdfs.append(p)
    if not all_pdfs:
        print("No PDFs found")
        exit(1)
    for pdf_path in all_pdfs:
        parsed = ResumeParser(pdf_path).parse()
        out_file = save_json(parsed, pdf_path, out_dir="parsed_resumes")
        print(f"Parsed {pdf_path.name} to {out_file}")
