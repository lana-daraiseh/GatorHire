import io
import re
from datetime import datetime
import pandas as pd
import streamlit as st
from pypdf import PdfReader
import pdfplumber
import json

import fitz  # PyMuPDF

# for language detection - can extend per our need
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
LINK_RE = re.compile(r'https?://\S+', re.I)
HR_RE = re.compile(r'(?m)^\s*([_\-–—=]{8,})\s*$')   # horizontal lines

# get layout of pdf (normal, 2 cols, etc)
def extract_text_layout_pymupdf(file_bytes: bytes):
    """
    Returns:
      full_text: str
      pages_lines: List[List[dict]] where each line dict has:
        {'text': str, 'x0': float, 'y0': float, 'x1': float, 'y1': float, 'font_size': float, 'is_bold': bool}
    """
    import statistics as stats

    full_text_parts = []
    pages_lines = []

    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            # use dict to get spans w font attributes
            blocks = page.get_text("dict")["blocks"]

            # flatten spans into line objects
            lines = []
            for b in blocks:
                if "lines" not in b: 
                    continue
                for l in b["lines"]:
                    span_texts = []
                    span_sizes = []
                    span_bolds = []
                    for s in l["spans"]:
                        txt = (s.get("text") or "").strip()
                        if not txt:
                            continue
                        span_texts.append(txt)
                        span_sizes.append(s.get("size", 0.0))
                        # bold check
                        span_bolds.append("Bold" in (s.get("font", "") or ""))

                    if not span_texts:
                        continue

                    line_text = " ".join(span_texts).strip()
                    x0 = min(c for c in [l["bbox"][0], l["bbox"][2]])
                    x1 = max(c for c in [l["bbox"][0], l["bbox"][2]])
                    y0 = min(c for c in [l["bbox"][1], l["bbox"][3]])
                    y1 = max(c for c in [l["bbox"][1], l["bbox"][3]])

                    lines.append({
                        "text": line_text,
                        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                        "font_size": stats.median(span_sizes) if span_sizes else 0.0,
                        "is_bold": any(span_bolds),
                    })

            # column aware order; sort by y, x
            lines.sort(key=lambda L: (round(L["y0"], 1), round(L["x0"], 1)))
            pages_lines.append(lines)

            full_text_parts.extend(l["text"] for l in lines)
            full_text_parts.append("\n")

    full_text = "\n".join(full_text_parts).strip()
    return full_text, pages_lines

# heading words to look out for
HEADING_WORDS = {"summary","skills","education","experience","projects","work experience","activities","leadership","coursework"}
# gets names
def parse_name_from_layout(pages_lines) -> str:
    if not pages_lines:
        return ""
    page0 = pages_lines[0]
    if not page0:
        return ""
    # consider only top 25% of the page by y
    ys = [ln["y1"] for ln in page0]
    if not ys: 
        return ""
    cutoff = min(ys) + 0.25*(max(ys)-min(ys))

    candidates = [ln for ln in page0 if ln["y0"] <= cutoff and 2 <= len(ln["text"].split()) <= 6]
    if not candidates:
        candidates = [ln for ln in page0 if 2 <= len(ln["text"].split()) <= 6]

    # largest font + not a section heading
    candidates.sort(key=lambda L: (L["font_size"], L["is_bold"]), reverse=True)
    for c in candidates:
        low = c["text"].strip().lower().rstrip(":")
        if low not in HEADING_WORDS and not any(w in low for w in ["gmail.com","linkedin.com","github.com","resume","curriculum vitae"]):
            return c["text"].strip()
    return ""

def parse_high_school_gpa(text: str) -> str:
    # look for GPA near hs / secondary
    candidates = []
    for m in re.finditer(r'(?i)(high\s*school|secondary).{0,80}?GPA[^0-9]{0,10}(\d(?:\.\d{1,2})?(?:\s*/\s*\d(?:\.\d{1,2})?)?)', text, flags=re.S):
        candidates.append(m.group(2))
    # generic gpa if clearly labeled and not obviously for college 
    if not candidates:
        for m in re.finditer(r'(?i)\bGPA\b[^0-9]{0,10}(\d(?:\.\d{1,2})?(?:\s*/\s*\d(?:\.\d{1,2})?)?)', text):
            candidates.append(m.group(1))
    # return most aesthetic
    return candidates[0] if candidates else ""

UNIV_LINE_RE = re.compile(
    r'(?im)^\s*(University|College|Institute|Polytechnic|Academy)[^\n]*$'
)

DEGREE_RE = re.compile(
    r'(?i)\b(Bachelor(?:’s|\'s)?|Master(?:’s|\'s)?|B\.?S\.?|BSc|M\.?S\.?|MSc|B\.?Eng|M\.?Eng|Ph\.?D\.?|Doctor(?:ate)?|Associate)\b[^\n,]*'
)
UNIV_NAME_INLINE_RE = re.compile(r'(?i)(University|College|Institute|Polytechnic|Academy)[^,\n]*')
def parse_degree_and_university(text: str) -> tuple[str, str]:
    degree = ""
    university = ""

    # first try: line that contains BOTH a degree word AND a university word
    for ln in text.splitlines():
        if DEGREE_RE.search(ln) and UNIV_NAME_INLINE_RE.search(ln):
            # degree = everything up to the university substring
            mu = UNIV_NAME_INLINE_RE.search(ln)
            university = mu.group(0).strip().rstrip(",")
            degree = ln[:mu.start()].strip().rstrip(",")
            if not degree:
                md = DEGREE_RE.search(ln)
                degree = md.group(0).strip()
            return degree, university

    # second try: finding them separately
    md = DEGREE_RE.search(text)
    if md:
        degree = md.group(0).strip()
        start = max(0, md.start() - 300); end = min(len(text), md.end() + 300)
        win = text[start:end]
        mu = UNIV_NAME_INLINE_RE.search(win)
        if mu:
            university = mu.group(0).strip().rstrip(",")
            return degree, university

    # third try: any university line anywhere
    mu3 = UNIV_NAME_INLINE_RE.search(text)
    if mu3: university = mu3.group(0).strip().rstrip(",")

    return degree, university

def parse_academic_status(text: str) -> dict:
    """major/minor/standing (e.g., 'Rising Senior')"""
    major = minor = standing = ""
    # look for majors/minors
    for ln in text.splitlines():
        m1 = re.search(r'(?i)\bmajor\b[:\s-]*([A-Za-z0-9 &/\-]+)', ln)
        if m1 and not major:
            major = m1.group(1).strip().rstrip(",.")
        m2 = re.search(r'(?i)\bminor\b[:\s-]*([A-Za-z0-9 &/\-]+)', ln)
        if m2 and not minor:
            minor = m2.group(1).strip().rstrip(",.")
    # standing check
    ms = re.search(r'(?i)\brising\s+(senior|junior|sophomore|freshman)\b', text)
    if ms:
        standing = f"Rising {ms.group(1).title()}"
    return {"major": major, "minor": minor, "academic_standing": standing}

def parse_skills_list(text: str) -> str:
    # skills / programming skills section
    m = re.search(r'(?is)^\s*(programming\s+skills|technical\s+skills|skills)\s*[:\n-]+(.*?)(?=^\s*[A-Z][A-Za-z ]{2,30}\s*$|\Z)', text, flags=re.M)
    if not m: 
        return ""
    sec = m.group(2)
    items = []
    for ln in sec.splitlines():
        ln = ln.strip().lstrip("•*-·").strip()
        if not ln: 
            continue
        # split by commas if the line is a list
        if "," in ln:
            items.extend([x.strip() for x in ln.split(",") if x.strip()])
        else:
            items.append(ln)
    # de-dupe + keep order
    seen, out = set(), []
    for it in items:
        k = it.lower()
        if k not in seen:
            seen.add(k); out.append(it)
    return ", ".join(out)[:800]

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
    # merge golang + go if found
    pretty = list(dict.fromkeys("Go" if x in ("Go", "Golang") else x for x in pretty))
    return ", ".join(pretty)

def parse_internships(text: str) -> list[dict]:
    edu_sec = slice_section(text, ("education","academic background"))
    scoped = text.replace(edu_sec, "") if edu_sec else text
    results = []
    lines = [ln.strip() for ln in scoped.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if re.search(r'(?i)\bintern\b', ln) and re.search(r'(?i)(software|data|ml|ai|developer|engineering|security|it)\b', ln):
            window = " ".join(lines[max(0, i-1):min(len(lines), i+3)])
            md = DATE_RANGE_RE.search(window)
            results.append({"line": ln, "dates": md.group(0) if md else ""})
    return results

EXP_HEAD_RE = re.compile(r'(?im)^\s*(work\s+experience|experience)\s*$')
NEXT_HEAD_RE = re.compile(r'(?im)^\s*[A-Z][A-Za-z ]{2,30}\s*$')
DATE_RANGE_RE = re.compile(
    rf'(?i)\b({MONTHS}\s+{YEAR}|\b{YEAR}\b)\s*[-–—]\s*(Present|{MONTHS}\s+{YEAR}|{YEAR})'
)

ALL_HEAD_RE = re.compile(r'(?im)^\s*(summary|objective|education|academic\s+background|programming\s+skills|technical\s+skills|skills|projects|experience|work\s+experience|activities|leadership|certifications|awards)\s*:?\s*$')

def slice_section(text: str, heading_names: tuple[str, ...]) -> str:
    head_pat = re.compile(r'(?im)^\s*(' + "|".join(re.escape(h) for h in heading_names) + r')\s*:?\s*$', re.M)
    m = head_pat.search(text)
    if not m:
        return ""
    start = m.end()
    stop_head = ALL_HEAD_RE.search(text, start)
    stop_rule = HR_RE.search(text, start)
    # choose whichever comes first
    stop = None
    if stop_head and stop_rule:
        stop = stop_head if stop_head.start() < stop_rule.start() else stop_rule
    else:
        stop = stop_head or stop_rule
    return text[start: stop.start()].strip() if stop else text[start:].strip()

ROLE_HINT = re.compile(r'(?i)\b(engineer|developer|intern|analyst|researcher|consultant|manager|assistant|associate|scientist|tutor|lead)\b')
ORGISH = re.compile(r'(?i)\b(inc\.?|llc|corp\.?|company|co\.|labs?|university|college|studio|technologies|systems|group|department)\b')
DATE_RANGE_RE = re.compile(rf'(?i)\b({MONTHS}\s+{YEAR}|{YEAR})\s*[-–—]\s*(Present|{MONTHS}\s+{YEAR}|{YEAR})')

def parse_experience(text: str) -> list[dict]:
    sec = slice_section(text, ("work experience","experience"))
    if not sec:
        return []
    raw_lines = [ln.strip() for ln in sec.splitlines()]

    # start new block when we see a date range OR potential header line
    blocks, cur = [], []
    def push():
        nonlocal cur, blocks
        if cur:
            blocks.append(cur); cur=[]

    for ln in raw_lines:
        if not ln:
            push(); continue
        headerish = DATE_RANGE_RE.search(ln) or ROLE_HINT.search(ln) or ("@" in ln) or (" - " in ln) or (" — " in ln)
        if headerish and cur:
            push()
        cur.append(ln)
    push()

    results = []
    for blk in blocks:
        header = " ".join(blk[:2])  # title/company often in 1–2 lines
        dates = ""
        md = DATE_RANGE_RE.search(header) or DATE_RANGE_RE.search(" ".join(blk))
        if md: dates = md.group(0)

        # split title/company w several separators
        title = company = ""
        parts = re.split(r'\s+[–—\-]\s+|\s@\s', header)
        if len(parts) >= 2:
            if ORGISH.search(parts[0]) and not ORGISH.search(parts[1]):
                company, title = parts[0].strip(", "), parts[1].strip(", ")
            else:
                title, company = parts[0].strip(", "), parts[1].strip(", ")
        else:
            title = blk[0].strip(", ")
            company = ""
            # if the next line looks like an organization, grab it
            if len(blk) > 1 and ORGISH.search(blk[1]):
                company = blk[1].strip(", ")

        bullets = []
        for ln in blk[1:]:
            if re.match(r'^[\u2022\-\u2219\*\·]+', ln):
                bullets.append(re.sub(r'^[\u2022\-\u2219\*\·]+\s*', '', ln))
            elif len(ln) > 30:  # long sentence-like line
                bullets.append(ln)
        results.append({"title": title, "company": company, "dates": dates, "bullets": bullets[:8]})
    return results

MMYYYY = r'([01]?\d)\s*/\s*(\d{4})'

def parse_graduation_date(text: str) -> str:
    # 1) explicit words like graduation / graduating 
    m = re.search(rf'(?i)\b(Graduation|Graduating|Expected|Anticipated)\b[^A-Za-z0-9]{{0,20}}(({MONTHS}\s+{YEAR})|{MMYYYY}|{YEAR})', text)
    if m:
        return m.group(2)

    # 2) scan education / degree lines for a right edge date
    for ln in text.splitlines():
        low = ln.lower()
        if any(k in low for k in ["education","bachelor","master","university","college","degree"]):
            me = re.search(rf'(({MONTHS}\s+{YEAR})|{MMYYYY}|{YEAR})\s*(?:–|-|to)?\s*(Present)?\s*$', ln.strip(), re.I)
            if me:
                return (me.group(1) + (" – Present" if me.group(4) else "")).strip()

    return ""

def parse_projects(text: str) -> list[dict]:
    sec = slice_section(text, ("projects",))
    if not sec:
        return []
    lines = [ln.strip() for ln in sec.splitlines() if ln.strip()]
    projects = []
    cur_name = ""
    cur_desc = []

    for ln in lines:
        if re.match(r'^[\u2022\-\*\·]', ln) or len(ln.split()) > 8:
            cur_desc.append(ln.lstrip("•*-· ").strip())
        else:
            # looks like a short title line
            if cur_name:
                projects.append({"name": cur_name, "description": " ".join(cur_desc)[:500]})
                cur_desc = []
            cur_name = ln.strip(":- ")
    if cur_name:
        projects.append({"name": cur_name, "description": " ".join(cur_desc)[:500]})
    return projects

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

PHONE_RE = re.compile(
    r'(\+?\d[\s\-\.\(\)]{0,3}\d{2,3}[\s\-\.\)]{0,3}\d{3}[\s\-\.\)]{0,3}\d{4})'
)

def normalize_phone(raw: str) -> str:
    digits = re.sub(r'\D', '', raw)
    if len(digits) == 11 and digits.startswith("1"):
        return "+1 " + f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    if len(digits) == 10:
        return f"({digits[0:3]}) {digits[3:6]}-{digits[6:]}"
    return raw.strip()

REFS_SPLIT_RE = re.compile(r'(?im)^\s*references\s*:?\s*$', re.M)

# created so that references phone numbers/emails dont count as users phone number/email
def _strip_references(text: str) -> str:
    # keep content before references 
    parts = REFS_SPLIT_RE.split(text)
    return parts[0] if parts else text

def guess_fields(text: str, pages_lines=None) -> dict:
    scoped = _strip_references(text)

    # first email / phone in order (not others!!)
    email = ""
    m_email = next(EMAIL_RE.finditer(scoped), None)
    if m_email:
        email = m_email.group(0)

    phone = ""
    m_phone = next(PHONE_RE.finditer(scoped), None)
    if m_phone:
        phone = normalize_phone(m_phone.group(0))

    links = LINK_RE.findall(scoped)
    linkedin = ", ".join(sorted({l for l in links if "linkedin.com" in l.lower()}))
    github = ", ".join(sorted({l for l in links if "github.com" in l.lower()}))
    other_links = ", ".join(sorted(set(links) - set(filter(None, linkedin.split(", "))) - set(filter(None, github.split(", ")))))[:300]

    # finding name; based on layout, else fallback
    name_guess = parse_name_from_layout(pages_lines) if pages_lines else ""
    if not name_guess:
        for ln in (ln.strip() for ln in scoped.splitlines() if ln.strip()):
            if EMAIL_RE.search(ln) or PHONE_RE.search(ln):
                continue
            if 2 <= len(ln.split()) <= 6 and ln.lower() not in HEADING_WORDS:
                name_guess = ln
                break

    # skills snippet - bounded to next heading
    skills_snip = ""
    m_sk = re.search(r'(?is)^\s*(programming\s+skills|technical\s+skills|skills)\s*[:\n-]+(.*?)(?=^\s*[A-Z][A-Za-z ]{2,30}\s*$|\Z)', scoped, flags=re.M)
    if m_sk:
        skills_snip = re.sub(r'\s+', ' ', m_sk.group(2)).strip()[:400]

    # education snippet
    edu_snip = ""
    m_edu = re.search(r'(?is)^\s*(education|academic\s+background)\s*[:\n-]*(.*?)(?=^\s*[A-Z][A-Za-z ]{2,30}\s*$|\Z)', scoped, flags=re.M)
    if m_edu:
        edu_snip = re.sub(r'\s+', ' ', m_edu.group(0)).strip()[:500]

    return {
        "name_guess": name_guess,
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "github": github,
        "other_links": other_links,
        "skills_snippet": skills_snip,
        "education_snippet": edu_snip,
        "char_count": len(scoped),
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

            text, pages_lines = extract_text_layout_pymupdf(file_bytes)
            if not text or len(text.strip()) < 20:
                text = extract_text_pdfplumber(file_bytes)
                pages_lines = []

            used_ocr = False
            if (not text or len(text.strip()) < 20) and use_ocr:
                ocr_text = ocr_text_from_pdf(file_bytes)
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    used_ocr = True
                    pages_lines = []

            fields = guess_fields(text, pages_lines)
            hs_gpa = parse_high_school_gpa(text)
            degree, university = parse_degree_and_university(text)
            languages = parse_programming_languages(text)
            interns = parse_internships(text)
            grad_when = parse_graduation_date(text)
            experiences = parse_experience(text)
            acad = parse_academic_status(text)
            skills_list = parse_skills_list(text)
            projects = parse_projects(text)

            rows.append({
                "file": uf.name,
                "pages": pages,
                "used_ocr": used_ocr,
                **fields,                    
                "hs_gpa": hs_gpa,
                "degree": degree,
                "university": university,
                **acad,                       
                "programming_languages": languages,
                "skills_list": skills_list,
                "internships_found": "; ".join(f"{i['line']} [{i['dates']}]" if i['dates'] else i['line'] for i in interns),
                "graduation_date": grad_when,
                "experience_count": len(experiences),
                "experience_json": json.dumps(experiences, ensure_ascii=False), 
                "projects_count": len(projects),
                "projects_json": json.dumps(projects, ensure_ascii=False),
            })


            with st.expander(f"Preview — {uf.name} ({pages if pages is not None else '?'} pages)"):
                st.text_area("Extracted text", text if text else "(no text found)", height=250)

            with st.expander(f"Experience — {uf.name}"):
                if experiences:
                    for i, ex in enumerate(experiences, 1):
                        st.markdown(f"**{i}. {ex['title']} @ {ex['company']}**  \n{ex['dates']}")
                        for b in ex["bullets"]:
                            st.markdown(f"- {b}")
                else:
                    st.write("No experience section detected.")

            with st.expander(f"Projects — {uf.name}"):
                if projects:
                    for p in projects:
                        st.markdown(f"**{p['name']}**")
                        if p['description']:
                            st.write(p['description'])
                else:
                    st.write("No projects section detected.")


        if rows:
            st.subheader("Parsed Summary")
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="parsed_resumes.csv", mime="text/csv")

     
    st.caption("Tip: If some PDFs show no text, they may be scanned. Enable OCR and ensure Tesseract & Poppler are installed.")
    # test OCR ^^

if __name__ == "__main__":
    main()
