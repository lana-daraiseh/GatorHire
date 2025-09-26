import io
import re
from datetime import datetime
import pandas as pd
import streamlit as st
from pypdf import PdfReader
import pdfplumber

import fitz  # PyMuPDF

# For language detection (extend as you wish)
LANG_KEYWORDS = {
    "python", "java", "c++", "c", "c#", "javascript", "typescript",
    "go", "golang", "rust", "ruby", "php", "swift", "kotlin", "matlab",
    "r", "sql", "scala", "perl", "dart", "bash", "shell", "html", "css"
}
MONTHS = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
YEAR = r"(20\d{2}|19\d{2})"

# OCR imports 
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_RE = re.compile(r'(\+?\d[\d\-\s\(\)]{7,}\d)')
LINK_RE = re.compile(r'https?://\S+', re.I)

# get layout of pdf (normal, 2 cols, etc)
def extract_text_layout_pymupdf(file_bytes: bytes) -> str:
    """
    extract text using PyMuPDF blocks to preserve reading order for
    multiple columns or complicated layouts
    """
    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            # get_text("blocks") returns a list of (x0, y0, x1, y1, "text", block_no, block_type, ...)
            blocks = page.get_text("blocks")
            # sort top-to-bottom then left-to-right
            blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
            # heuristic -cluster into cols if necessary
            # (for my resume) simple two col split by median x
            xs = [ (b[0]+b[2])/2 for b in blocks ]  # centers
            if xs:
                mid = sorted(xs)[len(xs)//2]
                left = [b for b in blocks if ((b[0]+b[2])/2) <= mid]
                right = [b for b in blocks if ((b[0]+b[2])/2) > mid]
                # decide if it really looks two columns (balance + separation)
                two_col = len(left) > 0 and len(right) > 0 and abs(len(left)-len(right)) < max(4, 0.3*len(blocks))
            else:
                two_col = False

            if two_col:
                # sort each col top 2 bottom; concat left then right
                left.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
                right.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
                ordered = left + right
            else:
                ordered = blocks

            for b in ordered:
                txt = (b[4] or "").strip()
                if txt:
                    text_parts.append(txt)
            text_parts.append("\n")
    return "\n".join(text_parts).strip()

def parse_high_school_gpa(text: str) -> str:
    # look for GPA near "High School" or "Secondary"
    candidates = []
    for m in re.finditer(r'(?i)(high\s*school|secondary).{0,80}?GPA[^0-9]{0,10}(\d(?:\.\d{1,2})?(?:\s*/\s*\d(?:\.\d{1,2})?)?)', text, flags=re.S):
        candidates.append(m.group(2))
    # generic GPA if clearly labeled and not obviously college-only
    if not candidates:
        for m in re.finditer(r'(?i)\bGPA\b[^0-9]{0,10}(\d(?:\.\d{1,2})?(?:\s*/\s*\d(?:\.\d{1,2})?)?)', text):
            candidates.append(m.group(1))
    # return most aesthetic
    return candidates[0] if candidates else ""

def parse_degree_and_university(text: str) -> tuple[str, str]:
    """
    Heuristics: capture degree line containing Bachelor/Master/BS/MS/etc
    and a nearby university/college/institute name.
    """
    degree_patterns = [
        r'(?i)\b(Bachelor(?:’s|\'s)?|Master(?:’s|\'s)?|B\.?S\.?|BSc|M\.?S\.?|MSc|B\.?Eng|M\.?Eng|Ph\.?D\.?|Doctor(?:ate)?|Associate)\b[^,\n]*',
    ]
    univ_patterns = r'(?i)\b(University|College|Institute|Polytechnic|Academy)\b[^\n,]*'

    degree = ""
    university = ""

    # find degree line first
    for pat in degree_patterns:
        m = re.search(pat, text)
        if m:
            degree = m.group(0).strip()
            # look around that position for a university name (+-200 chars)
            start = max(0, m.start() - 200)
            end = min(len(text), m.end() + 200)
            window = text[start:end]
            mu = re.search(univ_patterns, window)
            if mu:
                # get whole line containing the match
                line = window.split("\n")[0] if "\n" in window[:mu.start()] else window[mu.start():].split("\n")[0]
                university = line.strip()
            break

    # if degree missing, still try to get university anywhere
    if not university:
        mu2 = re.search(univ_patterns, text)
        if mu2:
            line = text[mu2.start():].split("\n", 1)[0]
            university = line.strip()

    return degree, university

def parse_programming_languages(text: str) -> str:
    # normalize + dedupe by lower-case
    found = set()
    for lang in LANG_KEYWORDS:
        # allow symbols like C++
        pattern = r'(?i)(?:^|[^A-Za-z0-9+])(' + re.escape(lang) + r')(?:[^A-Za-z0-9+]|$)'
        if re.search(pattern, text):
            found.add(lang)
    # capitalize langs
    pretty = []
    cap = {"python": "Python", "java":"Java", "c++":"C++", "c#":"C#", "c":"C", "go":"Go", "golang":"Go"}
    for f in sorted(found):
        pretty.append(cap.get(f, f.capitalize()))
    # merge "golang" + "Go" if both found
    pretty = list(dict.fromkeys("Go" if x in ("Go", "Golang") else x for x in pretty))
    return ", ".join(pretty)

def parse_internships(text: str) -> list[dict]:
    """
    find lines with 'Intern' and tech-y keywords. return small records.
    """
    results = []
    # split in 2 bullets / lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if re.search(r'(?i)\bintern\b', ln) and re.search(r'(?i)(software|data|ml|ai|developer|engineering|security|it)\b', ln):
            # try to capture date range on same or close lines
            window = " ".join(lines[max(0, i-1):min(len(lines), i+2)])
            md = re.search(rf'(?i)({MONTHS}\s+{YEAR})\s*[-–—]\s*(Present|{MONTHS}\s+{YEAR})', window)
            when = md.group(0) if md else ""
            results.append({"line": ln, "dates": when})
    return results

def parse_graduation_date(text: str) -> str:
    # look for Graduation / Expected / Anticipated key words
    m = re.search(rf'(?i)\b(Graduation|Expected|Anticipated)\b[^A-Za-z0-9]{{0,20}}({MONTHS}\s+{YEAR}|{YEAR})', text)
    if m:
        return m.group(2)
    #  look on degree lines
    m2 = re.search(rf'(?i)\b({MONTHS}\s+{YEAR}|{YEAR})\b.*\b(Bachelor|Master|B\.?S\.?|M\.?S\.?|BSc|MSc|Ph\.?D\.?)', text)
    if m2:
        return m2.group(1)
    return ""

def extract_text_pdfplumber(file_bytes: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text += t + "\n"
    return text

def ocr_text_from_pdf(file_bytes: bytes) -> str:
    # OCR fallback for scanned PDFs. requires pdf2image + pytesseract + Poppler + Tesseract
    if not OCR_AVAILABLE:
        return ""
    try:
        images = convert_from_bytes(file_bytes, dpi=300)
    except Exception:
        return ""
    chunks = []
    for img in images:
        try:
            chunks.append(pytesseract.image_to_string(img))
        except Exception:
            pass
    return "\n".join(chunks)

def guess_fields(text: str) -> dict:
    emails = EMAIL_RE.findall(text)
    phones = PHONE_RE.findall(text)
    links = LINK_RE.findall(text)

    # quick section based snippets -super heuristic
    skills_snip = ""
    m_sk = re.search(r'(?im)^\s*(skills|technical skills)\s*[:\n-]+(.*?)(\n[A-Z][^\n]{0,40}\n|\Z)', text)
    if m_sk:
        skills_snip = re.sub(r'\s+', ' ', m_sk.group(2)).strip()[:300]

    edu_snip = ""
    m_edu = re.search(r'(?is)(education|academic background).*?(?=(experience|projects|skills|certifications|$))', text)
    if m_edu:
        edu_snip = re.sub(r'\s+', ' ', m_edu.group(0)).strip()[:300]

    # name guess = first non-empty line that isn't just contact info
    name_guess = ""
    for ln in (ln.strip() for ln in text.splitlines() if ln.strip()):
        if EMAIL_RE.search(ln) or PHONE_RE.search(ln):
            continue
        if 2 <= len(ln.split()) <= 6:
            name_guess = ln
            break

    # get linkedin + github if present
    linkedin = ", ".join(sorted({l for l in links if "linkedin.com" in l.lower()}))
    github = ", ".join(sorted({l for l in links if "github.com" in l.lower()}))
    other_links = ", ".join(sorted(set(links) - set(linkedin.split(", ")) - set(github.split(", "))))[:300]

    return {
        "name_guess": name_guess,
        "emails": ", ".join(sorted(set(emails))),
        "phones": ", ".join(sorted({p.strip() for p in phones})),
        "linkedin": linkedin,
        "github": github,
        "other_links": other_links,
        "skills_snippet": skills_snip,
        "education_snippet": edu_snip,
        "char_count": len(text),
    }

def page_count(file_bytes: bytes):
    try:
        return len(PdfReader(io.BytesIO(file_bytes)).pages)
    except Exception:
        return None

def main():
    st.set_page_config(page_title="PDF Resume Parser", layout="wide")
    st.title("PDF Resume Parser")
    st.write("Upload one or more PDFs. We’ll extract text and pull several fields for their analysis.")

    use_ocr = st.checkbox(
        "Use OCR fallback for scanned PDFs (requires Tesseract + Poppler)",
        value=False,
        help="Enable this if you expect scanned/image-only PDFs. You must have Tesseract and Poppler installed on your system."
    )

    files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    rows = []

    if files:
        for uf in files:
            file_bytes = uf.read()
            pages = page_count(file_bytes)

            text = extract_text_layout_pymupdf(file_bytes)
            if not text or len(text.strip()) < 20:
                text = extract_text_pdfplumber(file_bytes)

            used_ocr = False

            if (not text or len(text.strip()) < 20) and use_ocr:
                ocr_text = ocr_text_from_pdf(file_bytes)
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    used_ocr = True

            fields = guess_fields(text)
            hs_gpa = parse_high_school_gpa(text)
            degree, university = parse_degree_and_university(text)
            languages = parse_programming_languages(text)
            interns = parse_internships(text)
            grad_when = parse_graduation_date(text)

            rows.append({
                "file": uf.name,
                "pages": pages,
                "used_ocr": used_ocr,
                **fields,
                "hs_gpa": hs_gpa,
                "degree": degree,
                "university": university,
                "programming_languages": languages,
                "internships_found": "; ".join(f"{i['line']} [{i['dates']}]" if i['dates'] else i['line'] for i in interns),
                "graduation_date": grad_when,
            })

            with st.expander(f"Preview — {uf.name} ({pages if pages is not None else '?'} pages)"):
                st.text_area("Extracted text", text if text else "(no text found)", height=250)

        if rows:
            st.subheader("Parsed Summary")
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="parsed_resumes.csv", mime="text/csv")

     
    st.caption("Tip: If some PDFs show no text, they may be scanned. Enable OCR and ensure Tesseract & Poppler are installed.")
    # figure this out ^

if __name__ == "__main__":
    main()
