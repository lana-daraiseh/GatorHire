#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Resume Parser with:
- DOCX extraction
- PDF native text extraction + OCR fallback (Tesseract)
- Robust section detection (incl. Research/Teaching)
- Improved education/experience parsing
- Date normalization to {start, end}
- Skills parsing with multiple delimiters and capitalization heuristics

Outputs: JSONL + CSV
"""

import os                      # paths, environment
import re                      # regex parsing
import json                    # JSONL output
import csv                     # CSV summary
import argparse                # CLI args
from typing import List, Dict, Optional, Tuple  # typing
from dataclasses import dataclass               # light structures
import fitz                    # PyMuPDF for PDF native text
from pdf2image import convert_from_path         # rasterize PDFs for OCR
import pytesseract            # OCR engine wrapper
import docx                   # python-docx (fallback reader)
import docx2txt               # primary DOCX text extractor
from PIL import Image         # images for OCR
import dateparser             # free-form date parsing to datetime
from datetime import datetime # normalization helpers

# --- Optional: if Poppler isn't on PATH, set it here (Windows zip path) ---
POPPLER_PATH = None  # e.g., r"C:\poppler-24.08.0\Library\bin" or leave None

# --- Point pytesseract to local Tesseract if you bundled it with the project ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder of this script
LOCAL_TESS = os.path.join(BASE_DIR, "Tesseract-OCR", "tesseract.exe")
if os.path.exists(LOCAL_TESS):                         # if bundled, use it
    pytesseract.pytesseract.tesseract_cmd = LOCAL_TESS

# --------------------------- Config / Heuristics -----------------------------

# Section headers we recognize (upper-cased match)
SECTION_HEADERS = [
    "SUMMARY", "PROFILE", "ABOUT",
    "SKILLS", "TECHNICAL SKILLS", "CORE COMPETENCIES",
    "EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", "EMPLOYMENT",
    "EDUCATION", "ACADEMICS",
    "PROJECTS",
    "CERTIFICATIONS", "LICENSES", "CERTS",
    "AWARDS", "HONORS", "PUBLICATIONS", "VOLUNTEER",
    "RESEARCH EXPERIENCE", "TEACHING EXPERIENCE"
]

# Regexes for common fields
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?:(?:(?:\+?\d{1,3})[\s\-\.])?(?:\(?\d{2,4}\)?[\s\-\.])?\d{3,4}[\s\-\.]\d{3,4})", re.X)
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.I)
LINKEDIN_RE = re.compile(r"\b(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/\S+\b", re.I)
GITHUB_RE = re.compile(r"\b(?:https?://)?(?:www\.)?github\.com/\S+\b", re.I)

# Degree/institution cues
DEGREE_WORDS = r"(Ph\.?D\.?|M\.?S\.?|MSc|MBA|MEng|B\.?S\.?|BSc|BEng|BA|MA|Associate|Diploma|Doctorate)"
DEGREE_RE = re.compile(rf"\b{DEGREE_WORDS}\b", re.I)

# Dates like "Jan 2019 - Feb 2020", "2017 - Present", etc. (hyphen or en-dash)
RANGE_SEP = r"[-\u2013]"
MONTH = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATE_RE = re.compile(
    rf"((?:{MONTH}\s+\d{{4}}|\d{{4}})\s*(?:{RANGE_SEP}\s*(?:{MONTH}\s+\d{{4}}|\d{{4}}|Present|Current))?)",
    re.I
)

# Location cue lines (best-effort)
LOCATION_HINTS = ["Address", "Location", "Based in", "Lives in"]

# Character density threshold to decide PDF native vs OCR
PDF_TEXT_DENSITY_THRESHOLD = 500  # chars/page

# Known org names to bias title/company split (extend as needed)
KNOWN_ORGS = [
    "Air Force Research Lab", "AFRL", "OWT Global", "PAR Government",
    "United States Air Forces in Europe", "USAFE-AFAFRICA", "St. John Fisher College",
    "Clarkson University", "University of Florida"
]


# ------------------------------- Utilities ----------------------------------

def normalize_text(s: str) -> str:
    """Normalize line endings, tabs, multi-spaces; collapse duplicate blanks."""
    s = s.replace("\r", "").replace("\t", " ")                    # unify whitespace
    lines = [re.sub(r"[ \u00A0]+", " ", ln).strip() for ln in s.split("\n")]  # per-line trim
    out = []
    for ln in lines:
        if not ln and (not out or not out[-1]):                   # drop duplicate empty lines
            continue
        out.append(ln)
    return "\n".join(out).strip()                                 # rejoin


def parse_single_date(s: str) -> Optional[datetime]:
    """Parse a single date token (e.g., 'Jan 2019' or '2017') -> datetime(YYYY,MM,1) or None."""
    if not s:
        return None
    s = s.strip().replace("Current", "Present")                   # unify wording
    if s.lower() == "present":
        return None                                               # represent as None for open range
    dt = dateparser.parse(s, settings={"PREFER_DAY_OF_MONTH": "first"})
    return dt                                                     # may be None if unparsable


def normalize_date_range(raw: str) -> Dict[str, Optional[str]]:
    """
    Convert a free-form date range into a dict:
    {'start':'YYYY-MM','end':'YYYY-MM' or 'Present' or None}
    """
    if not raw:
        return {"start": None, "end": None}
    parts = re.split(RANGE_SEP, raw)                              # split on hyphen/en-dash
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:                                                 # no dates found
        return {"start": None, "end": None}
    if len(parts) == 1:                                           # single date only
        start_dt = parse_single_date(parts[0])
        return {"start": start_dt.strftime("%Y-%m") if start_dt else None, "end": None}
    # two-sided range
    start_dt = parse_single_date(parts[0])
    end_raw = parts[1]
    if re.search(r"present|current", end_raw, re.I):
        end_iso = "Present"
    else:
        end_dt = parse_single_date(end_raw)
        end_iso = end_dt.strftime("%Y-%m") if end_dt else None
    start_iso = start_dt.strftime("%Y-%m") if start_dt else None
    # if reversed (rare), swap
    if start_dt and isinstance(end_iso, str) and end_iso != "Present":
        try:
            end_dt = datetime.strptime(end_iso, "%Y-%m")
            if end_dt < start_dt:
                start_iso, end_iso = end_iso, start_iso
        except Exception:
            pass
    return {"start": start_iso, "end": end_iso}


# ---------------------------- Text Extraction -------------------------------

def extract_docx(path: str) -> str:
    """Extract text from a .docx using docx2txt with python-docx fallback."""
    text = docx2txt.process(path) or ""                           # primary extractor
    if not text.strip():                                          # fallback if empty
        d = docx.Document(path)
        text = "\n".join([p.text for p in d.paragraphs])
    return normalize_text(text)


def extract_pdf_native(path: str) -> str:
    """Extract text from a PDF using PyMuPDF (native text)."""
    doc = fitz.open(path)                                         # open PDF
    texts = []
    for page in doc:                                              # iterate pages
        texts.append(page.get_text("text"))                       # text mode
    return normalize_text("\n".join(texts))


def extract_pdf_ocr(path: str, dpi: int = 300, tesseract_lang: str = "eng") -> str:
    """OCR a PDF: rasterize each page then run Tesseract; join results."""
    kwargs = {}
    if POPPLER_PATH:                                              # set poppler path if needed
        kwargs["poppler_path"] = POPPLER_PATH
    pages = convert_from_path(path, dpi=dpi, **kwargs)            # rasterize
    out = []
    for img in pages:                                             # each page image
        out.append(pytesseract.image_to_string(img, lang=tesseract_lang))
    return normalize_text("\n".join(out))


def read_any(path: str, threshold: int = PDF_TEXT_DENSITY_THRESHOLD) -> Tuple[str, str]:
    """Detect type and return (text, method): 'docx-native' | 'pdf-native' | 'pdf-ocr' | 'pdf-native-lowtext'."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":                                            # DOCX path
        return extract_docx(path), "docx-native"
    if ext == ".pdf":                                             # PDF path
        txt_native = extract_pdf_native(path)                     # native text first
        try:
            pages = fitz.open(path).page_count                    # page count
        except Exception:
            pages = 1
        density = len(txt_native) / max(1, pages)                 # chars per page
        if density > threshold:                                   # enough native text
            return txt_native, "pdf-native"
        txt_ocr = extract_pdf_ocr(path)                           # fallback OCR
        if len(txt_ocr) >= len(txt_native):                       # prefer richer
            return txt_ocr, "pdf-ocr"
        return txt_native, "pdf-native-lowtext"
    raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------- Parsing Helpers -------------------------------

def guess_name(text: str) -> Optional[str]:
    """
    Guess the candidate name:
    1) Look at first 6 non-empty lines, skip lines with obvious contacts.
    2) Prefer Title-Case 2 4 tokens.
    3) Fallback: first non-empty line that looks like Firstname Lastname pattern.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()] # non-empty lines
    top = lines[:6]                                               # look near top
    candidates = []                                               # collected names
    for ln in top:
        if EMAIL_RE.search(ln) or PHONE_RE.search(ln) or URL_RE.search(ln):
            continue
        if 3 <= len(ln) <= 80:
            tokens = [t for t in re.split(r"\s+", ln) if t]
            if 1 < len(tokens) <= 5:
                titleish = sum(1 for t in tokens if re.match(r"^[A-Z][a-z' \-]+$", t))
                if titleish >= max(2, len(tokens) - 1):
                    candidates.append(ln)
    if candidates:
        return candidates[0]
    # Fallback: First Last(-Last) pattern
    for ln in top:
        if re.match(r"^[A-Z][a-zA-Z' \-]+\s+[A-Z][a-zA-Z' \-]+(?:\s+[A-Z][a-zA-Z' \-]+)?$", ln):
            return ln
    return None


def split_sections(text: str) -> Dict[str, str]:
    """
    Split the text into sections using known headers.
    Returns: dict[HEADER] = body text
    """
    headers = "|".join([re.escape(h) for h in SECTION_HEADERS])   # build alt group
    header_re = re.compile(rf"^(?P<h>({headers}))\b[:\s\-]*$", re.I | re.M)  # header lines
    sections: Dict[str, str] = {}
    matches = list(header_re.finditer(text))                      # find all headers
    if not matches:
        sections["FULL"] = text                                   # no headers found
        return sections
    for i, m in enumerate(matches):
        start = m.end()                                           # start of section body
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        header = m.group("h").upper()                             # header in upper
        body = text[start:end].strip()                            # slice body
        sections[header] = body
    return sections


def extract_contact(text: str) -> Dict:
    """Extract emails, phones, links, optional top-of-doc location line."""
    emails = list(dict.fromkeys(EMAIL_RE.findall(text)))          # de-duped
    phones = list(dict.fromkeys([re.sub(r"[^\d+()\-\s]", "", p).strip()
                                 for p in PHONE_RE.findall(text)]))
    links_raw = list(dict.fromkeys(URL_RE.findall(text)))
    special = []
    for patt in (LINKEDIN_RE, GITHUB_RE):                         # pin important links first
        for m in patt.findall(text):
            if m not in special:
                special.append(m)
    links = list(dict.fromkeys(special + links_raw))
    location = None
    for ln in text.split("\n")[:15]:                              # near top only
        if any(h.lower() in ln.lower() for h in LOCATION_HINTS):
            location = ln.strip()
            break
    return {"emails": emails, "phones": phones, "links": links, "location": location}


def parse_skills(block: str) -> List[str]:
    """
    Parse 'Skills' blocks separated by commas/semicolons/newlines.
    Also split runs of capitalized tokens if separated by spaces only (e.g., 'MATLAB C# C++').
    """
    # First split on commas/semicolons/newlines
    raw = re.split(r"[,;\n]+", block)
    parts = []
    for r in raw:
        t = r.strip("  -*").strip()
        if not t:
            continue
        # If it looks like multiple tokens jammed with spaces, split further
        if " " in t and re.search(r"([A-Za-z#\+\.\-]+\s+){2,}", t):
            parts.extend([p for p in re.split(r"\s{2,}|\s(?=[A-Z][^a-z])", t) if p.strip()])
        else:
            parts.append(t)
    # De-duplicate while preserving order
    out, seen = [], set()
    for p in parts:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def match_or_none(patt: re.Pattern, text: str) -> Optional[str]:
    """Return first regex match string or None."""
    m = patt.search(text or "")
    return m.group(0) if m else None


def parse_education(block: str) -> List[Dict]:
    """
    Parse education chunks into degree, field (best-effort), institution, dates, and notes.
    Strategy:
      - Split on blank lines.
      - For each chunk: pull degree phrase, date range/single date, GPA/Graduated lines into notes.
      - Institution: prefer a line that contains 'University'/'College' or follows a degree line.
    """
    entries = []
    chunks = [c.strip() for c in re.split(r"\n\s*\n", block.strip()) if c.strip()]
    for ch in chunks:
        lines = [l.strip() for l in ch.split("\n") if l.strip()]
        joined = " ".join(lines)
        degree_m = DEGREE_RE.search(joined)
        degree = degree_m.group(0) if degree_m else None
        date_raw = match_or_none(DATE_RE, joined)
        dates = normalize_date_range(date_raw) if date_raw else {"start": None, "end": None}

        institution = None
        # Prefer a line with University/College
        for l in lines:
            if re.search(r"\b(University|College|Institute)\b", l, re.I):
                institution = l
                break
        if not institution:
            # Fallback: if second line isn't a graduated/GPA line, use it
            for l in lines[:3]:
                if not re.search(r"\b(GPA|Graduated|Expected)\b", l, re.I):
                    if l != degree:
                        institution = l
                        break

        # Notes keep everything (useful for GPAs etc.)
        entries.append({
            "institution": institution,
            "degree": degree,
            "field": None,
            "dates": dates,
            "notes": ch
        })
    return entries


def looks_like_title(s: str) -> bool:
    """Heuristic: detect job titles by keywords."""
    return bool(re.search(r"(engineer|developer|manager|director|lead|analyst|scientist|consultant|intern|liaison|assistant|subject matter expert|program manager|test engineer|field service)", s, re.I))


def parse_experience_header(header: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse an experience header into (title, company, location).
    Patterns handled:
      - 'Title, Company, City, ST'
      - 'Title   Company, City, ST'
      - 'Company   Title'
      - Simple single line (guess by keywords & KNOWN_ORGS).
    """
    h = header.strip()
    parts = re.split(r"\s+[ \-]\s+|,\s*", h)                      # split on dash or commas
    parts = [p for p in parts if p]
    title = company = location = None

    # If we have at least two parts, try to assign
    if len(parts) >= 2:
        # Bias: if any part matches known orgs, that's company
        org_idx = None
        for i, p in enumerate(parts):
            if any(org.lower() in p.lower() for org in KNOWN_ORGS):
                org_idx = i
                break

        if org_idx is not None:
            company = parts[org_idx]
            # Title likely adjacent to the org token
            if org_idx > 0 and looks_like_title(parts[org_idx - 1]):
                title = parts[org_idx - 1]
            elif org_idx + 1 < len(parts) and looks_like_title(parts[org_idx + 1]):
                title = parts[org_idx + 1]
        else:
            # No known orgs: pick first title-like piece as title; next as company
            for i, p in enumerate(parts):
                if looks_like_title(p):
                    title = p
                    # company candidate is the next non-title-like token
                    for q in parts[i+1:]:
                        if not looks_like_title(q):
                            company = q
                            break
                    break

        # Location: any remaining trailing token with comma or state-like pattern
        if len(parts) >= 3:
            loc_candidates = [p for p in parts[-2:] if re.search(r"[A-Z]{2}$|\b[A-Za-z]+\s*(?:City|NY|FL|Qatar|UAE|Jordan|Bahrain)\b", p)]
            if loc_candidates:
                location = loc_candidates[-1]

    # Fallbacks
    if not title and looks_like_title(h):
        title = h
    if not company and not looks_like_title(h):
        company = h

    return title, company, location


def parse_experience(block: str) -> List[Dict]:
    """
    Parse experience into list of jobs:
      - Split on blank lines
      - First line is the header; next lines are bullets
      - Detect dates anywhere in the chunk; normalize
      - Use header parser for title/company/location
    """
    entries = []
    chunks = [c.strip() for c in re.split(r"\n\s*\n", block.strip()) if c.strip()]
    for ch in chunks:
        lines = [l for l in ch.split("\n") if l.strip()]
        if not lines:
            continue
        header = lines[0]
        dates_raw = match_or_none(DATE_RE, ch)
        dates = normalize_date_range(dates_raw) if dates_raw else {"start": None, "end": None}
        title, company, location = parse_experience_header(header)
        bullets = []
        for ln in lines[1:]:
            ln_clean = ln.strip(" *- \t")
            if ln_clean:
                bullets.append(ln_clean)
        entries.append({
            "company": company,
            "title": title,
            "location": location,
            "dates": dates,
            "bullets": bullets
        })
    return entries


def parse_projects(block: str) -> List[Dict]:
    """Parse projects as {name, dates?, bullets[]} by paragraph."""
    projects = []
    for chunk in re.split(r"\n\s*\n", block.strip()):
        if not chunk.strip():
            continue
        lines = [l.strip() for l in chunk.split("\n") if l.strip()]
        title = lines[0] if lines else None
        dates_raw = match_or_none(DATE_RE, chunk)
        dates = normalize_date_range(dates_raw) if dates_raw else {"start": None, "end": None}
        bullets = [l.strip(" *- \t") for l in lines[1:]]
        projects.append({"name": title, "dates": dates, "bullets": bullets})
    return projects


def extract_summary(text: str, sections: Dict[str, str]) -> Optional[str]:
    """Prefer SUMMARY/PROFILE/ABOUT; otherwise first few non-contact lines."""
    for key in ("SUMMARY", "PROFILE", "ABOUT"):
        if key in sections and sections[key].strip():
            lines = sections[key].strip().split("\n")
            return "\n".join(lines[:6]).strip()
    head = []
    for ln in text.split("\n")[:15]:
        if EMAIL_RE.search(ln) or PHONE_RE.search(ln) or URL_RE.search(ln):
            continue
        if ln.strip():
            head.append(ln.strip())
        if len(head) >= 6:
            break
    return "\n".join(head).strip() if head else None


# ------------------------------ Main Parser ---------------------------------

def parse_resume(text: str, source_path: str, extraction_method: str) -> Dict:
    """End-to-end parsing of a single resume's text into structured fields."""
    name = guess_name(text)
    contact = extract_contact(text)
    sections = split_sections(text)

    # Skills
    skills = []
    for sk in ("SKILLS", "TECHNICAL SKILLS", "CORE COMPETENCIES"):
        if sk in sections:
            skills = parse_skills(sections[sk])
            if skills:
                break

    # Education
    education = parse_education(sections.get("EDUCATION", "")) if "EDUCATION" in sections else []

    # Experience (any of the experience headers)
    experience = []
    for ex in ("EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", "EMPLOYMENT", "RESEARCH EXPERIENCE", "TEACHING EXPERIENCE"):
        if ex in sections:
            experience.extend(parse_experience(sections[ex]))

    # Projects
    projects = parse_projects(sections.get("PROJECTS", "")) if "PROJECTS" in sections else []

    # Certifications
    certs = []
    for ce in ("CERTIFICATIONS", "LICENSES", "CERTS"):
        if ce in sections:
            certs = [l.strip(" *- \t") for l in sections[ce].split("\n") if l.strip()]
            if certs:
                break

    summary = extract_summary(text, sections)

    return {
        "source_path": source_path,
        "extraction_method": extraction_method,
        "name": name,
        "contact": contact,
        "summary": summary,
        "skills": skills,
        "education": education,
        "experience": experience,
        "projects": projects,
        "certifications": certs,
        "raw_text": text
    }


# ------------------------------- I/O / CLI ----------------------------------

def collect_files(root: str) -> List[str]:
    """Gather all .pdf and .docx under a folder (or a single file)."""
    out = []
    if os.path.isdir(root):
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in (".pdf", ".docx"):
                    out.append(os.path.join(dirpath, fn))
    else:
        out.append(root)
    return out


def write_jsonl(items: List[Dict], out_path: str) -> None:
    """Write one JSON object per line (JSONL)."""
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def write_csv(items: List[Dict], out_path: str) -> None:
    """Write a compact summary CSV for quick sanity checks."""
    fields = ["source_path", "extraction_method", "name", "emails", "phones", "links", "top_skill", "first_company", "first_title"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for it in items:
            emails = ", ".join(it.get("contact", {}).get("emails", []))
            phones = ", ".join(it.get("contact", {}).get("phones", []))
            links = ", ".join(it.get("contact", {}).get("links", []))
            top_skill = (it.get("skills") or [""])[0] if it.get("skills") else ""
            first_company = first_title = ""
            if it.get("experience"):
                first_company = it["experience"][0].get("company") or ""
                first_title = it["experience"][0].get("title") or ""
            w.writerow({
                "source_path": it.get("source_path", ""),
                "extraction_method": it.get("extraction_method", ""),
                "name": it.get("name", "") or "",
                "emails": emails,
                "phones": phones,
                "links": links,
                "top_skill": top_skill,
                "first_company": first_company,
                "first_title": first_title
            })


def main():
    """CLI entrypoint: read files, extract text (native/OCR), parse, and write outputs."""
    ap = argparse.ArgumentParser(description="Resume Parser (PDF+DOCX) with OCR fallback and normalized dates")
    ap.add_argument("input", help="File or folder of resumes")
    ap.add_argument("-o", "--out", default="parsed_resumes.jsonl", help="Output JSONL file")
    ap.add_argument("--csv", default="parsed_summary.csv", help="Output CSV summary file")
    ap.add_argument("--ocr-lang", default="eng", help="Tesseract language code")
    ap.add_argument("--pdf-density-threshold", type=int, default=PDF_TEXT_DENSITY_THRESHOLD, help="Chars/page threshold to decide OCR")
    args = ap.parse_args()

    threshold = args.pdf_density_threshold                          # threshold for native-vs-OCR
    paths = collect_files(args.input)                               # gather files

    results = []
    for p in paths:
        try:
            text, method = read_any(p, threshold)                   # extract text
            if method == "pdf-ocr":                                 # ensure OCR uses requested lang
                text = extract_pdf_ocr(p, tesseract_lang=args.ocr_lang)
            parsed = parse_resume(text, p, method)                  # parse to schema
            results.append(parsed)                                  # collect
            print(f"[OK] {method:16s}  {p}")                        # progress
        except Exception as e:
            print(f"[ERR] {p}: {e}")                                # error marker

    write_jsonl(results, args.out)                                  # JSONL output
    write_csv(results, args.csv)                                    # CSV summary
    print(f"\nWrote {len(results)} records to:\n  JSONL: {args.out}\n  CSV:   {args.csv}")


if __name__ == "__main__":
    main()
