import re
from typing import Dict, List


SECTION_HEADERS = [
    "summary", "objective", "experience", "work experience",
    "professional experience", "education", "projects",
    "certifications", "skills", "relevant coursework", 
    "activities and leadership", "honors and awards"
]


def extract_contact_info(text: str) -> Dict:
    """Extract basic contact info using regex."""
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    phone_pattern = r"\+?\d[\d\-\(\) ]{8,}\d"
    url_pattern = r"(https?:\/\/[^\s]+)"

    email = re.search(email_pattern, text)
    phone = re.search(phone_pattern, text)
    links = re.findall(url_pattern, text)

    # crude name guess = first line (improve later with NER)
    first_line = text.splitlines()[0].strip()

    return {
        "name": first_line if len(first_line.split()) <= 5 else "",
        "email": email.group(0) if email else "",
        "phone": phone.group(0) if phone else "",
        "links": links
    }


def split_sections(text: str) -> Dict[str, str]:
    """Split text into sections based on keywords."""
    sections = {}
    current_header = "other"
    sections[current_header] = []

    for line in text.splitlines():
        line_clean = line.strip().lower()
        if not line_clean:
            continue

        # check if line is a header
        if any(line_clean.startswith(h) for h in SECTION_HEADERS):
            current_header = line_clean
            sections[current_header] = []
        else:
            sections[current_header].append(line.strip())

    # join lines
    return {k: "\n".join(v) for k, v in sections.items()}


def extract_resume_structure(text: str) -> Dict:
    """Main function: extract structured info from resume text."""
    contact = extract_contact_info(text)
    sections = split_sections(text)

    # skill & certification extraction (expand later with NLP)
    certifications_text = sections.get("certifications", "")
    certifications = re.split(r"[,;\n]", certifications_text)
    certifications = [c.strip() for c in certifications if c.strip()]

    skills_text = sections.get("skills", "")
    skills = re.split(r"[,;\n]", skills_text)
    skills = [s.strip() for s in skills if s.strip()]

    return {
        "contact": contact,
        "summary": sections.get("summary", sections.get("objective", "")),
        "experience": sections.get("experience", sections.get("work experience", "")),
        "education": sections.get("education", ""),
        "projects": sections.get("projects", ""),
        "certifications": certifications,
        "skills": skills
    }


# test
if __name__ == "__main__":
    from tempParser import parse_resume 

    resume_path = "Christian Schneider Resume.docx"
    parsed = parse_resume(resume_path)
    structured = extract_resume_structure(parsed["text"])

    import json
    print(json.dumps(structured, indent=2))
