from __future__ import annotations
from typing import Any, Dict, Iterable, Optional
import argparse
import json
from pathlib import Path

#JSON output format
SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["full_name", "emails", "phones", "education", "experience"],
    "properties": {
        "full_name": {"type": "string"},
        "headline": {"type": "string"},
        "location": {"type": "string"},
        "emails": {"type": "array", "items": {"type": "string"}},
        "phones": {"type": "array", "items": {"type": "string"}},
        "links": {"type": "array", "items": {"type": "string"}},
        "skills": {"type": "array", "items": {"type": "string"}},
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["institution", "degree"],
                "properties": {
                    "institution": {"type": "string"},
                    "degree": {"type": "string"},
                    "field": {"type": "string"},
                    "gpa": {"type": ["number", "string", "null"]},
                    "start_date": {"type": ["string", "null"]},
                    "end_date": {"type": ["string", "null"]},
                    "ongoing": {"type": ["boolean", "null"]},
                },
                "additionalProperties": False,
            },
        },
        "experience": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["company", "title"],
                "properties": {
                    "company": {"type": "string"},
                    "title": {"type": "string"},
                    "location": {"type": "string"},
                    "start_date": {"type": ["string", "null"]},
                    "end_date": {"type": ["string", "null"]},
                    "ongoing": {"type": ["boolean", "null"]},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                    "tech": {"type": "array", "items": {"type": "string"}},
                    "impact_metrics": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
        },
        "projects": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "tech": {"type": "array", "items": {"type": "string"}},
                    "links": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
        },
        "certs": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are an expert resume parser. Extract facts only from the provided resume text. "
    "If a field is missing, use null or an empty array as appropriate. Output valid JSON that matches the provided JSON schema exactly. Do not include commentary or extra keys."
)

def _read_text_from_pdf(path: Path, *, max_pages: Optional[int] = None) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = reader.pages
    n = len(pages) if max_pages is None else min(len(pages), max_pages)
    texts: Iterable[str] = (pages[i].extract_text() or "" for i in range(n))
    return "\n\n".join(texts).strip()

def _read_text_from_docx(path: Path) -> str:
    import docx
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs).strip()

def _read_text_from_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()

def _ocr_pdf_to_text(path: Path) -> str:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
    images = convert_from_path(str(path))
    texts = []
    for img in images:
        if not isinstance(img, Image.Image):
            img = img.convert("RGB")
        texts.append(pytesseract.image_to_string(img))
    return "\n\n".join(texts).strip()

def _ingest(path: Path, *, ocr: bool = False) -> str:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    ext = path.suffix.lower()
    if ext == ".pdf":
        text = _read_text_from_pdf(path)
        if ocr and len(text) < 200:
            text = _ocr_pdf_to_text(path)
    elif ext == ".docx":
        text = _read_text_from_docx(path)
    elif ext in {".txt", ".text"}:
        text = _read_text_from_txt(path)
    else:
        raise ValueError("Unsupported file type. Use .pdf, .docx, or .txt")
    if not text or len(text) < 20:
        raise ValueError("No usable text extracted. Try --ocr for scanned PDFs.")
    return text

#ollama model call
def _call_ollama_for_json(resume_text: str, *, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434") -> str:
    import requests
    user = (
        f"Extract JSON with this schema and ONLY return JSON.\n\n"
        f"SCHEMA:\n{json.dumps(SCHEMA)}\n\n"
        f"RESUME:\n{resume_text}"
    )
    prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user}\n<|assistant|>"
    payload = {"model": model, "prompt": prompt, "options": {"temperature": 0.0}, "stream": False}
    r = requests.post(f"{base_url.rstrip('/')}/api/generate", json=payload, timeout=180)
    r.raise_for_status()
    content = r.json().get("response", "").strip()
    if not content:
        raise RuntimeError("Empty response from Ollama.")
    return content

#public api
def extract_resume_info_simple(input_path: str | Path, *, model: str = "llama3.1:8b", ocr: bool = False) -> Dict[str, Any]:
    """Return a dict of resume fields extracted by a local Ollama model."""
    text = _ingest(Path(input_path), ocr=ocr)
    raw = _call_ollama_for_json(text, model=model)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(raw[start:end+1])
        else:
            raise ValueError("Model did not return valid JSON.")
    for key in ["emails", "phones", "links", "skills", "projects", "certs", "education", "experience"]:
        if key not in data or data[key] is None:
            data[key] = [] if key not in {"education", "experience"} else []
    for edu in data.get("education", []):
        gpa = edu.get("gpa")
        if isinstance(gpa, str):
            try:
                edu["gpa"] = float(gpa)
            except ValueError:
                edu["gpa"] = None
    return data

#CLI
def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract resume JSON using local Ollama model.")
    p.add_argument("--in", dest="input_path", required=True, help="Path to resume (.pdf/.docx/.txt)")
    p.add_argument("--model", default="llama3.1:8b", help="Ollama model (default: llama3.1:8b)")
    p.add_argument("--ocr", action="store_true", help="Enable OCR fallback for scanned PDFs")
    return p.parse_args()


def main(argv: Optional[list[str]] = None) -> int:
    args = _args()
    try:
        data = extract_resume_info_simple(args.input_path, model=args.model, ocr=args.ocr)
    except Exception as e:
        print(f"Error: {e}")
        return 2
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
