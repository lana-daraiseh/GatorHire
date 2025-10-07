# PDF Resume Parser (Streamlit)

A lightweight Streamlit app to parse PDF resumes and extract structured info:
**name, email/phone, degree & university, graduation date, GPA,
skills (raw + cleaned), programming languages, experience (title/company/dates/bullets),
internships, and projects.** Works on single- and multi-column resumes, with an optional
**OCR** fallback for scanned PDFs.

---

## Features

- **Layout-aware extraction** (PyMuPDF) for better reading order and heading detection.
- **Sectionized parsing**:
  - **Experience** → multiple entries with title, company, dates, and bullets.
  - **Education** → degree + university (even on the **same line**), graduation date.
  - **Skills** → raw section snippet **and** a cleaned, de-duplicated list.
  - **Programming languages** → keyword pass across the whole document.
  - **Projects** → name + short description.
  - **Internships** → derived from Experience first; textual fallback if needed.
- **First contact only** → keeps only the first **email** and **phone** (ignores “References” section).
- **CSV export**

---

## Requirements

**Python:** 3.9–3.12 • **OS:** Windows / macOS / Linux

### Python packages

Install:
```bash
pip install streamlit pandas pypdf pdfplumber PyMuPDF
# optional (only needed for OCR fallback):
pip install pdf2image pytesseract Pillow
