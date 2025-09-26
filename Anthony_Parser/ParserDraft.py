import sys, re, json, argparse, pathlib
from typing import List, Dict, Any, Tuple


import fitz  
import pdfplumber
import docx2txt
import phonenumbers
from dateutil import parser as dateparser

import cv2
import pytesseract

# NLP
import spacy
from spacy.tokens import Doc

BULLETS = r"[•·◦▪■◆▶►\-–—•]"
SKILL_SPLIT_RE = re.compile(r"\s*(?:{}|\||,|\u25C6|\u2022)\s*".format(BULLETS))
SECTION_HEADERS = [
    "summary", "objective", "skills", "technical skills", "projects", "experience",
    "work experience", "education", "certifications", "references", "activities",
    "leadership", "awards", "publications"
]

ADDRESS_WORDS = {"drive","dr","road","rd","street","st","avenue","ave","boulevard","blvd","lane","ln","way","court","ct"}

NON_PERSON_PHRASES = {
    "Summa Cum Laude","Cum Laude","Magna Cum Laude",
    "LinkedIn","Aurora","Ollama","React","TypeScript",
    "LLM","AI","RAG","AP Exams","AP Computer Science Principles"
}

NON_ORG_PAT = re.compile(r"@|https?://", re.I)
GENERIC_ORG_WORDS = {"ai","data","services","team","project","projects","office","media","specialist","school","counselor"}
SCHOOL_SUBJECTS = {"computer","science","principles","exams"}

DATE_LIKE = re.compile(r"(?i)\b(?:\d{4}|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|spring|summer|fall|winter)\b")

def load_spacy() -> Any:
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return spacy.blank("en")

NLP = load_spacy()

def read_docx(path: str) -> str:
    return docx2txt.process(path) or ""

def read_pdf_text(path: str) -> Tuple[str, bool, int]:
    """
    Try structured extract (PyMuPDF), fallback to pdfplumber, then OCR (images).
    Return (text, ocr_used, pages_ocr)
    """
    text = []
    ocr_used = False
    ocr_pages = 0

    #PyMuPDF
    try:
        doc = fitz.open(path)
        for page in doc:
            page_text = page.get_text("text")
            if page_text.strip():
                text.append(page_text)
            else:
                pageno = page.number
                with pdfplumber.open(path) as pdf:
                    if pageno < len(pdf.pages):
                        t = pdf.pages[pageno].extract_text() or ""
                        if t.strip():
                            text.append(t)
                            continue
                pix = page.get_pixmap(matrix=fitz.Matrix(2,2))  # upscale for OCR
                img = cv2.imdecode(
                    memoryview(pix.tobytes("png")),
                    cv2.IMREAD_COLOR
                )
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                    t = pytesseract.image_to_string(gray)
                    if t.strip():
                        text.append(t)
                        ocr_used = True
                        ocr_pages += 1
        doc.close()
    except Exception:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                text.append(t)

    return ("\n".join(text), ocr_used, ocr_pages)

def read_input(path: str) -> Tuple[str, bool, int]:
    p = pathlib.Path(path)
    if p.suffix.lower() in [".docx", ".doc"]:
        return (read_docx(path), False, 0)
    else:
        return read_pdf_text(path)

def normalize_text(s: str) -> List[str]:
    s = s.replace("\u2013","-").replace("\u2014","-").replace("\u25C6","◆").replace("\u2022","•")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in s.splitlines()]
    return [ln for ln in lines if ln]

def sectionize(lines: List[str]) -> Dict[str, List[str]]:
    sections: Dict[str,List[str]] = {}
    cur = "header"
    sections[cur] = []
    for ln in lines:
        low = ln.lower()
        header_hit = None
        for h in SECTION_HEADERS:
            if low.startswith(h):
                header_hit = h
                break
        if header_hit:
            cur = header_hit
            sections.setdefault(cur, [])
            rest = ln[len(header_hit):].strip(" :-")
            if rest:
                sections[cur].append(rest)
        else:
            sections.setdefault(cur, [])
            sections[cur].append(ln)
    return sections

def extract_emails(text: str) -> List[str]:
    return list(dict.fromkeys(re.findall(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", text)))

def extract_links(text: str) -> List[str]:
    urls = re.findall(r"https?://\S+", text)
    return list(dict.fromkeys(urls))

def extract_phones(text: str) -> List[str]:
    out = []
    for match in phonenumbers.PhoneNumberMatcher(text, "US"):
        out.append(phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.NATIONAL))
    return list(dict.fromkeys(out))

def guess_name_from_header(header_lines: List[str]) -> str:
    for ln in header_lines[:6]:
        if ln.isupper() and len(ln.split()) >= 2 and len(ln) <= 60:
            return ln.title()

    for ln in header_lines[:6]:
        tokens = [t for t in re.split(r"\s+", ln) if t]
        caps = [t for t in tokens if re.match(r"^[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?$", t)]
        if len(caps) >= 2:
            return " ".join(caps[:3])
    return ""

def split_skills(sk_lines: List[str]) -> List[str]:
    out = []
    for ln in sk_lines:
        parts = [p.strip(" -") for p in SKILL_SPLIT_RE.split(ln) if p.strip(" -")]
        out.extend(parts)

    seen = set()
    uniq = []
    for s in out:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq

def parse_education(sec: Dict[str,List[str]]) -> List[Dict[str,str]]:
    edu = []
    edulines = sec.get("education", [])
    bucket = []
    for ln in edulines:
        if DATE_LIKE.search(ln) or ln.isupper():
            bucket.append(ln)
        else:
            bucket.append(ln)
 
    i = 0
    while i < len(bucket):
        line = bucket[i]
        if line.isupper() and len(line) > 3:
            school = line.title()
            degree = ""
            dates = ""

            j = i+1
            grabbed = 0
            while j < len(bucket) and grabbed < 3:
                if DATE_LIKE.search(bucket[j]):
                    dates = bucket[j]
                elif not degree and not bucket[j].isupper():
                    degree = bucket[j]
                grabbed += 1
                j += 1
            edu.append({"school": school, "degree": degree, "dates": dates})
            i = j
        else:
            i += 1
    return edu

def chunk_experience_lines(xplines: List[str]) -> List[List[str]]:
    """Group experience lines into entries: start a new entry when a line looks like a new role/org."""
    chunks = []
    cur = []
    for ln in xplines:
        if not cur:
            cur = [ln]
            continue

        if re.match(r"^[A-Z].{0,80}$", ln) or re.match(r".+\s-\s.+", ln):

            chunks.append(cur)
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        chunks.append(cur)
    return chunks

def clean_org_guess(text_block: str, known_orgs: List[str]) -> str:

    low = text_block.lower()
    for org in known_orgs:
        if org and org.lower() in low:
            return org

    m = re.match(r"^([A-Z][A-Za-z0-9& .’'-]{2,60}?)(?:\s*[:-]|$)", text_block)
    if m:
        cand = m.group(1).strip()
        # reject if generic
        if len(cand.split()) > 1 and all(w.lower() not in GENERIC_ORG_WORDS for w in cand.split()):
            return cand
    return ""

def parse_experience(sec: Dict[str,List[str]], filtered_ents: Dict[str,List[str]]) -> List[Dict[str,Any]]:
    xplines = sec.get("experience", [])
    # group lines into chunks
    chunks = chunk_experience_lines(xplines)
    known_orgs = filtered_ents.get("ORG", []) + filtered_ents.get("GPE", [])
    out = []
    for ch in chunks:
        if not ch: 
            continue
        head = ch[0]
        body = " ".join(ch)
        title = ""
        if " - " in head:
            title = head.split(" - ", 1)[1].strip()
        else:
            m = re.match(r"^(.{3,80}?)(?:\:|$)", head)
            if m:
                title = m.group(1).strip()
        dates = ""
        md = DATE_LIKE.search(body)
        if md:
            dates = md.group(0)
        org = clean_org_guess(body, known_orgs)
        bullets = []
        if len(ch) > 1:
            bullets = [" ".join(ch[1:]).strip()]
        out.append({
            "title": title or head,
            "organization": org,
            "dates": dates,
            "bullets": bullets
        })
    for e in out:
        if not e["organization"] and re.search(r"\b(project|personal|independent)\b", " ".join(e["bullets"]), re.I):
            e["organization"] = "Personal Project"
    return out

def filter_ner(doc: Doc) -> Dict[str,List[str]]:
    ents = {"PERSON": [], "ORG": [], "GPE": [], "DATE": [], "FAC": []}
    for ent in doc.ents:
        label = ent.label_
        text = ent.text.strip()
        # normalize whitespace
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue

        if label == "PERSON":
            toks = text.split()
            if len(toks) >= 2 and all(re.match(r"^[A-Z][a-z]+", t) for t in toks):
                if text not in NON_PERSON_PHRASES:
                    ents["PERSON"].append(text)

        elif label == "ORG":
            if NON_ORG_PAT.search(text):
                continue  # drop emails/URLs
            low = text.lower()
            toks = [t for t in re.findall(r"[A-Za-z]+", text)]
            if any(t in SCHOOL_SUBJECTS for t in toks):
                continue
            if all(t in GENERIC_ORG_WORDS for t in toks):
                continue
            ents["ORG"].append(text)

        elif label == "GPE":
            ents["GPE"].append(text)

        elif label == "DATE":
            # drop weird stuff
            if re.search(r"\d{3}[-–]\d{3,4}", text):
                continue
            ents["DATE"].append(text)

        elif label == "FAC":
            toks = text.lower().split()
            if len(toks) >= 2 and any(w in ADDRESS_WORDS for w in toks):
                ents["FAC"].append(text)

    for k in ents:
        seen = set()
        uniq = []
        for v in ents[k]:
            key = v.lower()
            if key not in seen:
                seen.add(key)
                uniq.append(v)
        ents[k] = uniq
    return ents


def extract_contact_from_header(sections: Dict[str,List[str]], filtered_ner: Dict[str,List[str]]) -> Dict[str,Any]:
    header = sections.get("header", [])[:8]  # only the very top area
    header_text = "\n".join(header)
    emails = extract_emails(header_text)
    phones = extract_phones(header_text)
    links = extract_links(header_text)

    name = ""
    if filtered_ner.get("PERSON"):
        # Choose person that appears in header
        cand = [n for n in filtered_ner["PERSON"] if any(n.lower() in ln.lower() for ln in header)]
        name = cand[0] if cand else filtered_ner["PERSON"][0]
    if not name:
        name = guess_name_from_header(header)

    return {
        "name": name,
        "emails": emails,
        "phones": phones,
        "links": links
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="Path to resume (PDF or DOCX)")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    raw_text, ocr_used, pages_ocr = read_input(args.file)
    lines = normalize_text(raw_text)
    sections = sectionize(lines)

    doc = NLP("\n".join(lines)) if hasattr(NLP, "pipe_names") and "ner" in NLP.pipe_names else None
    filtered_ents = filter_ner(doc) if doc else {"PERSON": [], "ORG": [], "GPE": [], "DATE": [], "FAC": []}

    contact = extract_contact_from_header(sections, filtered_ents)

    # Skills
    skills = split_skills(sections.get("skills", []))

    # Education
    education = parse_education(sections)

    # Experience 
    experience = parse_experience(sections, filtered_ents)

    #don’t let references’ emails/phones leak into contact
    out = {
        "source_file": pathlib.Path(args.file).name,
        "meta": {"ocr_used": ocr_used, "pages_ocr": pages_ocr},
        "contact": contact,
        "sections_present": [k for k,v in sections.items() if v],
        "skills": skills,
        "experience": experience,
        "education": education,
        "raw_sections": {k: v for k, v in sections.items() if v},
        "ner_preview": filtered_ents  
    }

    print(json.dumps(out, indent=2 if args.pretty else None, ensure_ascii=False))

if __name__ == "__main__":
    main()
#test
    #py parser.py "Anthony Franzino Resume 2025.pdf" --pretty
