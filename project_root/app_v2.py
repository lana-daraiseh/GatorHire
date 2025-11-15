# Future annotations for forward references
from __future__ import annotations  # enable postponed evaluation of annotations

# Stdlib imports
import json  # JSON utilities
import os    # filesystem utilities
import math  # math helpers
from typing import Any, Dict, List, Tuple  # type hints

# Third-party and framework imports
from flask import (
    Flask,
    render_template_string,
    request,
    redirect,
    url_for,
    jsonify,
)  # Flask tools
import pandas as pd  # DataFrame operations
from sklearn.feature_extraction.text import TfidfVectorizer  # text vectorization
from sklearn.cluster import KMeans  # clustering
from slugify import slugify  # safe filenames and keys

# Local resume parsing APIs
from resume_batch_parser import batch_parse_resumes, process_resume  # batch + single parse

# Defaults and constants
DEFAULT_FOLDER = "Resumes"                # default resume folder
DEFAULT_OUTPUT_CSV = "parsed_resumes.csv" # default CSV output path
DEFAULT_LOG_FILE = "parser_log.log"       # default parser log path
DEFAULT_LIMIT = 20                        # default max files to parse per run
DEFAULT_WORKERS = 1                       # default concurrency

# Flask app setup
app = Flask(__name__)                                 # init Flask app
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024   # 25MB upload limit
UPLOAD_DIR = "uploads"                                # uploads directory
os.makedirs(UPLOAD_DIR, exist_ok=True)                # ensure uploads dir exists

# In-memory model store for quick inference
MODEL: Dict[str, Any] = {
    "vectorizer": None,            # fitted TfidfVectorizer
    "kmeans": None,                # fitted KMeans
    "terms": None,                 # vocabulary terms list
    "labels": None,                # training labels per doc
    "df_index_map": None,          # map from local doc index -> df row
    "skill_vocab": {},             # global skill frequency
    "cluster_skill_vocab": {},     # per-cluster skill frequency
    "title_labels": [],            # common job titles per cluster
}

# Empty chart and cluster placeholders for first render
DEFAULT_CHARTS: Dict[str, Any] = {
    "skills": {"labels": [], "values": []},             # top skills chart
    "companies": {"labels": [], "values": []},          # top companies chart
    "titles": {"labels": [], "values": []},             # top job titles chart
    "schools": {"labels": [], "values": []},            # top schools chart
    "gpa_dist": {"labels": [], "values": []},           # GPA distribution chart
    "skills_per_resume": {"labels": [], "values": []},  # skills-per-resume distribution
}
DEFAULT_CLUSTERS = {"summaries": []}                   # default cluster summaries


# Small helpers
# Guard dashboard payload shape to avoid template or JS errors
def _safe_dashboard_payload(charts, clusters):
    if not isinstance(charts, dict):
        charts = DEFAULT_CHARTS  # fallback charts
    else:
        # ensure all expected keys exist
        for key in DEFAULT_CHARTS:
            charts.setdefault(key, {"labels": [], "values": []})
    if not isinstance(clusters, dict) or "summaries" not in clusters:
        clusters = DEFAULT_CLUSTERS  # fallback clusters
    return charts, clusters  # always safe structures


# String sanitization helper
def _s(x: Any) -> str:
    return x.strip() if isinstance(x, str) else ""  # trimmed str or empty


# Tail helper for logs so UI only shows recent lines
def _tail_log(path: str, max_lines: int = 400) -> str:
    if not path or not os.path.exists(path):
        return ""  # missing file
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:])  # last N lines
    except Exception:
        return ""  # read failure


# JSON coercion helper
def _parse_json_field(x: Any) -> Any:
    """
    Take a cell that might be:
    - already a list/dict
    - a JSON-like string
    - something else

    and try to normalize it into a list/dict where appropriate.
    """
    if isinstance(x, (list, dict)):
        return x  # already structured
    if not isinstance(x, str):
        return x  # leave non-strings alone
    txt = x.strip()
    if not txt or not (txt.startswith("{") or txt.startswith("[")):
        return x  # not JSON text
    try:
        return json.loads(txt)
    except Exception:
        return x  # fall back to original string on parse failure


def _ensure_list(x: Any) -> List[Any]:
    """
    Normalize a field into a list:

    - list -> list
    - JSON string representing a list -> parsed list
    - anything else -> []
    """
    if isinstance(x, list):
        return x
    parsed = _parse_json_field(x)
    return parsed if isinstance(parsed, list) else []


# Skill token normalization for counting and matching
def _norm_skill_token(s: str) -> str:
    s = (s or "").strip()  # trim
    if not s:
        return ""  # empty guard
    s = s.replace("/", " /")  # separate slashed items
    return " ".join(s.split()).lower()  # collapse spaces and lowercase


# Compact education cell for table view (first 3 entries)
def _compact_education(cell: str) -> str:
    items = _ensure_list(cell)      # parse cell JSON/list or []
    out: List[str] = []             # pieces
    for it in items[:3]:
        it = it or {}
        school = _s(it.get("school", ""))  # school
        degree = _s(it.get("degree", ""))  # degree
        year = _s(it.get("year", ""))      # year
        piece = ", ".join([p for p in [school, degree, year] if p])  # joined
        if piece:
            out.append(piece)      # add piece
    return " • ".join(out)         # bullet join


# Turn a comma skills string into a deduped, capped list
def _skills_list(cell: str, cap: int = 12) -> List[str]:
    toks = [t.strip() for t in (cell or "").split(",") if t.strip()]  # split
    seen = set()                                                      # dedupe set
    out = []                                                          # output list
    for t in toks:
        low = t.lower()                                               # casefold
        if low not in seen:
            seen.add(low)                                             # mark
            out.append(t)                                             # keep original case
    return out[:cap]                                                  # cap list


# Build up to N table rows for dashboard
def _table_rows(df: pd.DataFrame, rows_to_show: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []            # output rows
    if df is None or df.empty:
        return rows                            # empty guard
    n = max(1, int(rows_to_show or 1))         # limit rows
    for _, r in df.head(n).iterrows():         # first N rows
        cl_raw = r.get("cluster", -1)          # raw cluster value
        try:
            cl = int(cl_raw)                   # int cluster
        except Exception:
            cl = -1                            # fallback
        rows.append({
            "filename": r.get("filename", ""),                         # filename
            "name": r.get("name", ""),                                 # parsed name
            "email": r.get("email", ""),                               # parsed email
            "phone": r.get("phone", ""),                               # parsed phone
            "cluster": ("" if cl < 0 else cl),                         # cluster or blank
            "skills_list": _skills_list(r.get("skills", "")),          # skills list
            "education_short": _compact_education(r.get("education", "")),  # compact edu
        })
    return rows  # dashboard table rows


# Insight builder: charts + clusters from CSV
def build_insights_from_csv(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Read the parsed resumes CSV and compute:

    - skills chart
    - company chart
    - title chart
    - school chart
    - GPA distribution
    - skills-per-resume distribution
    - cluster summaries (via KMeans on text corpus)
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame(), DEFAULT_CHARTS, DEFAULT_CLUSTERS  # missing file -> empties

    df = pd.read_csv(csv_path, dtype=str).fillna("")  # read as strings

    # Skills aggregation for chart
    all_skills: List[str] = []  # normalized tokens

    def split_skills(x: str) -> List[str]:
        return [s.strip() for s in (x or "").split(",") if s.strip()]  # split comma skills

    if "skills" in df.columns:
        for cell in df["skills"].tolist():
            for tok in split_skills(cell):
                norm = _norm_skill_token(tok)  # normalize
                if norm:
                    all_skills.append(norm)    # collect

    skills_freq: Dict[str, int] = {}  # frequency map
    for s in all_skills:
        skills_freq[s] = skills_freq.get(s, 0) + 1  # count

    top_skills = sorted(skills_freq.items(), key=lambda kv: (-kv[1], kv[0]))[:20]  # top 20
    skills_chart = {"labels": [k for k, _ in top_skills], "values": [v for _, v in top_skills]}  # chart payload

    # Company / title / school aggregation
    companies: List[str] = []  # company names
    titles: List[str] = []     # job titles
    schools: List[str] = []    # school names

    if "experience" in df.columns:
        for cell in df["experience"].tolist():
            exp_list = _ensure_list(cell)
            for item in exp_list:
                item = item or {}
                comp = _s(item.get("company", ""))
                title = _s(item.get("title", ""))
                if comp:
                    companies.append(comp)
                if title:
                    titles.append(title)

    if "education" in df.columns:
        for cell in df["education"].tolist():
            edu_list = _ensure_list(cell)
            for item in edu_list:
                item = item or {}
                sch = _s(item.get("school", ""))
                if sch:
                    schools.append(sch)

    def _freq_map(values: List[str]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for v in values:
            out[v] = out.get(v, 0) + 1
        return out

    company_freq = _freq_map(companies)
    title_freq = _freq_map(titles)
    school_freq = _freq_map(schools)

    top_companies = sorted(company_freq.items(), key=lambda kv: (-kv[1], slugify(kv[0] or "")))[:20]
    top_titles = sorted(title_freq.items(), key=lambda kv: (-kv[1], slugify(kv[0] or "")))[:20]
    top_schools = sorted(school_freq.items(), key=lambda kv: (-kv[1], slugify(kv[0] or "")))[:20]

    companies_chart = {"labels": [k for k, _ in top_companies], "values": [v for _, v in top_companies]}
    titles_chart = {"labels": [k for k, _ in top_titles], "values": [v for _, v in top_titles]}
    schools_chart = {"labels": [k for k, _ in top_schools], "values": [v for _, v in top_schools]}

    # GPA distribution (bucketed)
    gpa_buckets: Dict[str, int] = {
        "<3.0": 0,
        "3.0–3.49": 0,
        "3.5–3.79": 0,
        "3.8–4.0": 0,
    }  # GPA buckets

    def _extract_row_gpa(row: pd.Series) -> float | None:
        gpa_value = row.get("gpa", "")  # direct GPA column
        if not gpa_value:
            edu_list = _ensure_list(row.get("education", ""))  # fallback to education JSON
            if edu_list:
                first = edu_list[0] or {}
                gpa_value = first.get("gpa") or first.get("GPA")
        if not gpa_value:
            return None
        try:
            return float(str(gpa_value).split()[0])
        except Exception:
            return None

    for _, row in df.iterrows():
        g = _extract_row_gpa(row)
        if g is None or g <= 0:
            continue
        if g < 3.0:
            gpa_buckets["<3.0"] += 1
        elif g < 3.5:
            gpa_buckets["3.0–3.49"] += 1
        elif g < 3.8:
            gpa_buckets["3.5–3.79"] += 1
        else:
            gpa_buckets["3.8–4.0"] += 1

    gpa_chart = {
        "labels": list(gpa_buckets.keys()),
        "values": list(gpa_buckets.values()),
    }  # GPA chart payload

    # Skills-per-resume distribution (how dense skills sections are)
    skills_count_buckets: Dict[str, int] = {
        "0": 0,
        "1–5": 0,
        "6–10": 0,
        "11–20": 0,
        "21+": 0,
    }  # skills count buckets

    if "skills" in df.columns:
        for cell in df["skills"].tolist():
            count = len(split_skills(cell))  # skills per resume
            if count == 0:
                skills_count_buckets["0"] += 1
            elif count <= 5:
                skills_count_buckets["1–5"] += 1
            elif count <= 10:
                skills_count_buckets["6–10"] += 1
            elif count <= 20:
                skills_count_buckets["11–20"] += 1
            else:
                skills_count_buckets["21+"] += 1

    skills_per_resume_chart = {
        "labels": list(skills_count_buckets.keys()),
        "values": list(skills_count_buckets.values()),
    }  # skills-per-resume chart payload

    # Build text corpus for clustering
    texts: List[str] = []     # documents
    index_map: List[int] = [] # doc->df index mapping

    for ix, row in df.iterrows():
        chunks: List[str] = []  # textual pieces

        # main fields
        chunks.append(row.get("summary", ""))               # summary text
        chunks.append(row.get("relevant_classwork", ""))    # course text

        # experience fields
        exp_list = _ensure_list(row.get("experience", ""))
        for item in exp_list:
            item = item or {}
            chunks.append(_s(item.get("title", "")))    # job title
            chunks.append(_s(item.get("company", "")))  # company
            bullets = item.get("bullets", []) or []     # bullet list
            if isinstance(bullets, list):
                chunks.extend([_s(b) for b in bullets])  # add bullets

        # projects fields
        proj_list = _ensure_list(row.get("projects", ""))
        for item in proj_list:
            item = item or {}
            chunks.append(_s(item.get("name", "")))         # project name
            chunks.append(_s(item.get("description", "")))  # project desc

        # research fields
        res_list = _ensure_list(row.get("research", ""))
        for item in res_list:
            item = item or {}
            chunks.append(_s(item.get("title", "")))        # research title
            chunks.append(_s(item.get("description", "")))  # research desc

        # education fields
        edu_list = _ensure_list(row.get("education", ""))
        for item in edu_list:
            item = item or {}
            chunks.append(_s(item.get("school", "")))       # school
            chunks.append(_s(item.get("degree", "")))       # degree

        text = "\n".join([c for c in chunks if isinstance(c, str)])  # join to doc
        if isinstance(text, str) and len(text.split()) > 2:          # min length
            texts.append(text)                                       # keep
            index_map.append(ix)                                     # map

    clusters_dict = {"summaries": []}  # default summaries
    df = df.copy()                     # prevent side effects
    df["cluster"] = -1                 # default cluster label

    # Clustering if enough documents
    if len(texts) >= 2:
        k = min(6, max(2, int(round(math.sqrt(len(texts))))))          # heuristic k
        vec = TfidfVectorizer(max_features=4000, stop_words="english") # vectorizer
        X = vec.fit_transform(texts)                                    # features
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)       # model
        labels = kmeans.fit_predict(X)                                  # assignments

        for local_i, df_i in enumerate(index_map):
            df.at[df_i, "cluster"] = int(labels[local_i])               # set label

        terms = vec.get_feature_names_out()                             # vocab
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]    # top term idx

        summaries = []  # build summaries per cluster
        for i in range(k):
            size = int((labels == i).sum())                             # cluster size
            top_terms = [terms[ind] for ind in order_centroids[i, :10]] # top 10 terms
            summaries.append({"cluster": int(i), "size": size, "top_terms": top_terms})
        clusters_dict = {"summaries": summaries}                         # final dict

    charts = {
        "skills": skills_chart,
        "companies": companies_chart,
        "titles": titles_chart,
        "schools": schools_chart,
        "gpa_dist": gpa_chart,
        "skills_per_resume": skills_per_resume_chart,
    }  # all charts bundle

    return df, charts, clusters_dict  # return insights


# Model training helpers
# Train TF-IDF + KMeans, build skill/title vocabularies, store in MODEL
def train_models_from_df(df: pd.DataFrame | None):
    """
    Train / reset the clustering models used by the analyzer.

    Passing None or an empty DataFrame will clear the in-memory model.
    """
    global MODEL  # mutate module-level cache

    if df is None or df.empty:
        MODEL.update({
            "vectorizer": None,
            "kmeans": None,
            "terms": None,
            "labels": None,
            "df_index_map": None,
            "skill_vocab": {},
            "cluster_skill_vocab": {},
            "title_labels": [],
        })  # clear model state
        return

    texts = []      # training docs
    index_map = []  # doc->df row
    for ix, row in df.iterrows():
        chunks = []                                                  # gather fields
        chunks.append(row.get("summary", ""))                        # summary
        chunks.append(row.get("relevant_classwork", ""))             # coursework

        exp_list = _ensure_list(row.get("experience", ""))           # experience
        for item in exp_list:
            item = item or {}
            chunks.append(_s(item.get("title", "")))                 # title
            chunks.append(_s(item.get("company", "")))               # company
            bullets = item.get("bullets", []) or []                  # bullets
            if isinstance(bullets, list):
                chunks.extend([_s(b) for b in bullets])              # add bullets

        proj_list = _ensure_list(row.get("projects", ""))           # projects
        for item in proj_list:
            item = item or {}
            chunks.append(_s(item.get("name", "")))                 # project name
            chunks.append(_s(item.get("description", "")))          # project desc

        res_list = _ensure_list(row.get("research", ""))            # research
        for item in res_list:
            item = item or {}
            chunks.append(_s(item.get("title", "")))                # research title
            chunks.append(_s(item.get("description", "")))          # research desc

        edu_list = _ensure_list(row.get("education", ""))           # education
        for item in edu_list:
            item = item or {}
            chunks.append(_s(item.get("school", "")))               # school
            chunks.append(_s(item.get("degree", "")))               # degree

        text = "\n".join([c for c in chunks if isinstance(c, str)])  # document
        if isinstance(text, str) and len(text.split()) > 2:          # min length
            texts.append(text)                                       # keep
            index_map.append(ix)                                     # map

    if not texts:
        # No usable text -> clear the model so analyzer gracefully refuses.
        MODEL.update({
            "vectorizer": None,
            "kmeans": None,
            "terms": None,
            "labels": None,
            "df_index_map": None,
            "skill_vocab": {},
            "cluster_skill_vocab": {},
            "title_labels": [],
        })
        return

    vec = TfidfVectorizer(max_features=4000, stop_words="english")  # vectorizer
    X = vec.fit_transform(texts)                                     # features
    k = min(8, max(2, int(round(math.sqrt(len(texts))))))            # heuristic k
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)        # model
    labels = kmeans.fit_predict(X)                                   # assignments

    MODEL["vectorizer"] = vec                    # store vectorizer
    MODEL["kmeans"] = kmeans                     # store model
    MODEL["terms"] = vec.get_feature_names_out().tolist()  # terms
    MODEL["labels"] = labels                     # labels per doc
    MODEL["df_index_map"] = index_map            # index map

    # Build skill frequencies (global and per cluster)
    skill_freq: Dict[str, int] = {}             # global freq
    cluster_skill: Dict[int, Dict[str, int]] = {i: {} for i in range(k)}  # per-cluster freq

    def add_skill(s, cid):
        s = _norm_skill_token(s)                  # normalize
        if not s:
            return                                # skip empties
        skill_freq[s] = skill_freq.get(s, 0) + 1  # global count
        cluster_skill[cid][s] = cluster_skill[cid].get(s, 0) + 1  # cluster count

    if "skills" in df.columns:
        for local_i, df_i in enumerate(index_map):
            cid = int(labels[local_i])                    # cluster id
            cell = df.iloc[df_i].get("skills", "")        # skills cell
            for tok in [t.strip() for t in (cell or "").split(",") if t.strip()]:
                add_skill(tok, cid)                       # update counts

    MODEL["skill_vocab"] = skill_freq             # save global vocab
    MODEL["cluster_skill_vocab"] = cluster_skill  # save cluster vocab

    # Aggregate common titles per cluster
    title_labels: List[List[str]] = [[] for _ in range(k)]  # titles per cluster
    if "experience" in df.columns:
        for local_i, df_i in enumerate(index_map):
            cid = int(labels[local_i])                                      # cluster id
            exp_list = _ensure_list(df.iloc[df_i].get("experience", ""))    # exp list
            for item in exp_list:
                item = item or {}
                t = _s(item.get("title", ""))                               # title
                if t:
                    title_labels[cid].append(t)                             # collect

    MODEL["title_labels"] = [
        [t for t, _ in sorted({t: title_labels[i].count(t) for t in title_labels[i]}.items(),
                              key=lambda kv: -kv[1])[:5]]
        for i in range(k)
    ]  # top-5 titles per cluster


# Template (single-page with three tabs)
TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Resume Parser Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
  :root{
    --bg:#020617;
    --card:#020617;
    --card-soft:#020617;
    --muted:#9ca3af;
    --txt:#e5e7eb;
    --accent:#38bdf8;
    --accent-soft:rgba(56,189,248,.08);
    --border:#1f2937;
    --danger:#f97373;
  }
  *{ box-sizing:border-box; }
  body{
    margin:0;
    font-family:Inter, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial;
    background:radial-gradient(circle at top,#0b1120 0,#020617 55%,#000 100%);
    color:var(--txt);
  }
  .wrap{ max-width:1200px; margin:0 auto; padding:24px; }
  .h1{ font-size:28px; font-weight:800; letter-spacing:0.2px; }
  .muted{ color:var(--muted); }
  .grid{ display:grid; gap:16px; }
  .grid-2{ grid-template-columns: 1fr 1fr; }
  .grid-3{ grid-template-columns: repeat(3,minmax(0,1fr)); }
  .card{
    background:var(--card);
    border:1px solid var(--border);
    border-radius:16px;
    padding:16px;
    box-shadow:0 18px 45px rgba(15,23,42,.75);
  }
  .btn{
    background:var(--accent);
    color:#020617;
    padding:10px 14px;
    border:none;
    border-radius:999px;
    font-weight:600;
    cursor:pointer;
    font-size:13px;
  }
  .btn-outline{
    background:transparent;
    color:var(--txt);
    border:1px solid var(--border);
  }
  .btn:disabled{ opacity:0.6; cursor:not-allowed; }
  input, select{
    width:100%;
    background:#020617;
    color:var(--txt);
    border:1px solid #1f2937;
    padding:10px 12px;
    border-radius:10px;
    font-size:13px;
  }
  label{
    font-weight:600;
    font-size:11px;
    color:#9ca3af;
    text-transform:uppercase;
    letter-spacing:0.06em;
  }
  table{ width:100%; border-collapse:collapse; font-size:13px; table-layout:fixed; }
  th, td{
    border-bottom:1px solid #111827;
    padding:8px 8px;
    text-align:left;
    vertical-align:middle;
    word-wrap:break-word;
  }
  th{ color:#9ca3af; font-weight:600; }
  tr{ height:1px; }
  code{
    background:#020617;
    color:#a7f3d0;
    padding:2px 6px;
    border-radius:6px;
    font-size:11px;
  }
  .pill{
    display:inline-block;
    padding:3px 8px;
    background:#020617;
    border:1px solid #1f2937;
    border-radius:999px;
    margin:2px 4px 2px 0;
    font-size:11px;
    line-height:1.3em;
    white-space:nowrap;
  }
  .tabs{
    display:flex;
    gap:8px;
    margin-top:16px;
    margin-bottom:16px;
    align-items:center;
  }
  .tab{
    font-size:13px;
    padding:6px 14px;
    border-radius:999px;
    border:1px solid transparent;
    background:transparent;
    color:var(--muted);
    text-decoration:none;
  }
  .tab.active{
    background:var(--accent-soft);
    border-color:var(--accent);
    color:var(--txt);
  }
  .tab:hover{ border-color:var(--border); }
  textarea.log-view{
    width:100%;
    min-height:260px;
    background:#020617;
    color:var(--txt);
    border-radius:10px;
    border:1px solid #111827;
    font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size:11px;
    padding:10px;
    resize:vertical;
    white-space:pre;
  }
  .badge{
    display:inline-flex;
    align-items:center;
    gap:4px;
    padding:3px 10px;
    border-radius:999px;
    border:1px solid #1f2937;
    background:#020617;
    font-size:11px;
    color:#9ca3af;
  }
  .badge-dot{
    width:7px;
    height:7px;
    border-radius:50%;
    background:var(--accent);
  }
  .badge-danger{ border-color:var(--danger); color:var(--danger); }
  .badge-dot-danger{ background:var(--danger); }
  .flex-between{
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:8px;
  }
  .analysis-grid{
    display:grid;
    gap:12px;
  }
  @media (min-width:900px){
    .analysis-grid{ grid-template-columns: minmax(0,1.4fr) minmax(0,1fr); }
  }
  .analysis-card-title{ font-size:14px; font-weight:600; margin-bottom:4px; }
  .analysis-list{ margin:4px 0 0 0; padding-left:18px; font-size:13px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="flex-between">
      <div>
        <div class="h1">GatorHire</div>
        <div class="muted" style="font-size:12px;">
          Batch parsing, clustering, and resume analysis in one place.
        </div>
      </div>
      <div class="badge">
        <span class="badge-dot"></span>
        Parsed rows in CSV: {{ parsed_rows }}
      </div>
    </div>

    <div class="tabs">
      <a href="{{ url_for('parser') }}" class="tab {% if active_tab == 'parser' %}active{% endif %}">Parser</a>
      <a href="{{ url_for('training_insights') }}" class="tab {% if active_tab == 'insights' %}active{% endif %}">Training &amp; Insights</a>
      <a href="{{ url_for('resume_analyzer') }}" class="tab {% if active_tab == 'analyzer' %}active{% endif %}">Resume Analyzer</a>
    </div>

    {% if message %}
      <div class="card" style="margin-bottom:12px;">{{ message }}</div>
    {% endif %}

    {# TAB 1: PARSER                                                       #}
    {% if active_tab == 'parser' %}

    <form class="card" method="post" action="{{ url_for('run_parse') }}">
      <div class="grid grid-2">
        <div>
          <label>Resumes folder</label>
          <input name="folder_path" value="{{ folder_path }}" />
        </div>
        <div>
          <label>Output CSV</label>
          <input name="output_csv" value="{{ output_csv }}" />
        </div>
        <div>
          <label>Log file</label>
          <input name="log_file" value="{{ log_file }}" />
        </div>
        <div>
          <label>Limit</label>
          <input name="limit" type="number" min="1" value="{{ limit }}" />
        </div>
        <div>
          <label>Max workers</label>
          <input name="max_workers" type="number" min="1" max="32" value="{{ max_workers }}" />
        </div>
      </div>
      <div style="margin-top:12px; display:flex; gap:8px;">
        <button class="btn" type="submit">Run Batch Parse</button>
        <a class="btn btn-outline" href="{{ url_for('parser') }}">Refresh view</a>
      </div>
    </form>

    <div class="card" style="margin-top:16px;">
      <div class="flex-between" style="margin-bottom:8px;">
        <div>
          <div style="font-weight:600; font-size:14px;">Parser Log Output</div>
          <div class="muted" style="font-size:12px;">This shows the most recent log lines from parser runs.</div>
        </div>
        <div style="display:flex; gap:8px;">
          <button class="btn btn-outline" type="button" onclick="refreshLog()">Refresh log</button>
          <span class="badge"><span class="badge-dot"></span> File: {{ log_file }}</span>
        </div>
      </div>
      <textarea id="logText" class="log-view" readonly>{{ log_contents }}</textarea>
    </div>

    <div class="card" style="margin-top:16px;">
      <div class="flex-between" style="margin-bottom:8px;">
        <div>
          <div style="font-weight:600; font-size:14px;">Parsed Resumes</div>
          <div class="muted" style="font-size:12px;">Showing rows from {{ output_csv }}.</div>
        </div>
        <div style="display:flex; gap:6px; align-items:center; font-size:12px;">
          <span>Rows to show:</span>
          <form method="get" action="{{ url_for('parser') }}" style="display:inline-flex; gap:6px; align-items:center;">
            <input name="rows" type="number" min="1" max="{{ max_table_rows }}" value="{{ rows_to_show }}" style="width:70px;" />
            <span class="muted" style="font-size:11px;">Max: {{ max_table_rows }}</span>
            <button class="btn btn-outline" type="submit" style="padding-inline:10px; font-size:11px;">Apply</button>
          </form>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>Filename</th>
            <th>Name</th>
            <th>Email</th>
            <th>Phone</th>
            <th>Cluster</th>
            <th>Skills</th>
            <th>Education (compact)</th>
          </tr>
        </thead>
        <tbody>
          {% for row in table_rows %}
          <tr>
            <td><code>{{ row.filename }}</code></td>
            <td>{{ row.name }}</td>
            <td>{{ row.email }}</td>
            <td>{{ row.phone }}</td>
            <td>{{ row.cluster }}</td>
            <td>
              {% for s in row.skills_list %}<span class="pill">{{ s }}</span>{% endfor %}
            </td>
            <td>{{ row.education_short }}</td>
          </tr>
          {% endfor %}
          {% if not table_rows %}
          <tr><td colspan="7" class="muted">No rows available.</td></tr>
          {% endif %}
        </tbody>
      </table>
    </div>

    {% endif %}

    {# TAB 2: TRAINING & INSIGHTS                                         #}
    {% if active_tab == 'insights' %}

    <form class="card" method="post" action="{{ url_for('train') }}">
      <div class="flex-between" style="margin-bottom:10px;">
        <div>
          <div style="font-weight:600; font-size:14px;">Train Models</div>
          <div class="muted" style="font-size:12px;">
            Uses the current CSV to build clusters and skills vocabulary for the analyzer.
          </div>
        </div>
        {% if model_ready %}
          <span class="badge"><span class="badge-dot"></span> Models ready</span>
        {% else %}
          <span class="badge badge-danger"><span class="badge-dot badge-dot-danger"></span> Not trained</span>
        {% endif %}
      </div>
      <div class="grid grid-2">
        <div>
          <label>Train from CSV</label>
          <input name="csv_path" value="{{ output_csv }}" />
        </div>
      </div>
      <div style="margin-top:12px; display:flex; gap:8px;">
        <button class="btn" type="submit">Train Models</button>
        <form method="post" action="{{ url_for('reset_models') }}" style="display:inline;">
          <!-- nested form is invalid HTML, so we keep reset button separate below -->
        </form>
      </div>
    </form>

    <form class="card" method="post" action="{{ url_for('reset_models') }}" style="margin-top:12px;">
      <div class="flex-between">
        <div>
          <div style="font-weight:600; font-size:14px;">Reset Analyzer Models</div>
          <div class="muted" style="font-size:12px;">
            Clears in-memory models and wipes charts &amp; cluster summaries until you retrain.
          </div>
        </div>
        <button class="btn btn-outline" type="submit">Reset Models</button>
      </div>
    </form>

    <div class="grid grid-2" style="margin-top:16px;">
      <div class="card">
        <h3 style="margin:4px 0 12px;">Top Skills</h3>
        <canvas id="skillsChart"></canvas>
      </div>
      <div class="card">
        <h3 style="margin:4px 0 12px;">Top Companies</h3>
        <canvas id="companiesChart"></canvas>
      </div>
    </div>

    <div class="grid grid-2" style="margin-top:16px;">
      <div class="card">
        <h3 style="margin:4px 0 12px;">Top Job Titles</h3>
        <canvas id="titlesChart"></canvas>
      </div>
      <div class="card">
        <h3 style="margin:4px 0 12px;">Top Schools</h3>
        <canvas id="schoolsChart"></canvas>
      </div>
    </div>

    <div class="grid grid-2" style="margin-top:16px;">
      <div class="card">
        <h3 style="margin:4px 0 12px;">GPA Distribution</h3>
        <canvas id="gpaChart"></canvas>
      </div>
      <div class="card">
        <h3 style="margin:4px 0 12px;">Skills per Resume</h3>
        <canvas id="skillsPerResumeChart"></canvas>
      </div>
    </div>

    <div class="card" style="margin-top:16px;">
      <h3 style="margin:4px 0 12px;">Cluster Summaries (KMeans)</h3>
      <div class="muted" style="margin-bottom:8px; font-size:12px;">
        Each resume is labeled by a cluster; top keywords per cluster summarize themes.
      </div>
      <div id="clusterSummary"></div>
    </div>

    {% endif %}

    {# TAB 3: RESUME ANALYZER                                             #}
    {% if active_tab == 'analyzer' %}

    <form class="card" method="post" action="{{ url_for('resume_analyzer') }}" enctype="multipart/form-data">
      <div class="flex-between" style="margin-bottom:12px;">
        <div>
          <div style="font-weight:600; font-size:14px;">Analyze a Resume</div>
          <div class="muted" style="font-size:12px;">
            Upload a single resume and compare it to trained clusters. Train models on the
            Training &amp; Insights tab first.
          </div>
        </div>
        {% if model_ready %}
          <span class="badge"><span class="badge-dot"></span> Analyzer ready</span>
        {% else %}
          <span class="badge badge-danger"><span class="badge-dot badge-dot-danger"></span> Train models first</span>
        {% endif %}
      </div>
      <div class="grid grid-2">
        <div>
          <label>Upload resume (PDF/DOCX)</label>
          <input type="file" name="resume" accept=".pdf,.docx" required />
        </div>
        <div>
          <label>Target role / internship keywords</label>
          <input name="target" placeholder="e.g., cybersecurity intern, data analyst, embedded systems" />
        </div>
      </div>
      <div style="margin-top:12px; display:flex; gap:8px;">
        <button class="btn" type="submit" {% if not model_ready %}disabled{% endif %}>Analyze Resume</button>
        <a class="btn btn-outline" href="{{ url_for('resume_analyzer') }}">Clear</a>
      </div>
    </form>

    {% if feedback %}
    <div class="card" style="margin-top:16px;">
      <div style="font-weight:600; font-size:14px; margin-bottom:4px;">
        Analysis Summary
      </div>
      <div class="muted" style="font-size:12px; margin-bottom:10px;">
        Cluster = {{ feedback.cluster }}, role match score =
        {% if feedback.role_score is not none %}{{ "%.3f"|format(feedback.role_score) }}{% else %}N/A{% endif %}.
        Suggested roles:
        {% if feedback.likely_titles %}
          {{ feedback.likely_titles|join(", ") }}.
        {% else %}
          N/A.
        {% endif %}
        {% if feedback.missing_skills %}
          Missing core skills: {{ feedback.missing_skills|join(", ") }}.
        {% endif %}
      </div>

      <div class="analysis-grid">
        <div class="card" style="background:#020617; border-radius:12px;">
          <div class="analysis-card-title">Profile &amp; Contact</div>
          <div style="font-size:13px;">
            <strong>Name:</strong> {{ feedback.parsed_contact.name or "N/A" }}<br/>
            <strong>Email:</strong> {{ feedback.parsed_contact.email or "N/A" }}<br/>
            {% if feedback.gpa_display %}
              <strong>GPA:</strong> {{ feedback.gpa_display }}<br/>
            {% endif %}
          </div>
          {% if feedback.section_counts %}
          <div style="margin-top:10px; font-size:12px;" class="muted">
            Sections detected:
            <ul class="analysis-list">
              <li>Experience entries: {{ feedback.section_counts.experience }}</li>
              <li>Project entries: {{ feedback.section_counts.projects }}</li>
              <li>Research entries: {{ feedback.section_counts.research }}</li>
              <li>Education entries: {{ feedback.section_counts.education }}</li>
            </ul>
          </div>
          {% endif %}
        </div>

        <div class="card" style="background:#020617; border-radius:12px;">
          <div class="analysis-card-title">Quick Gaps</div>
          <ul class="analysis-list">
            {% if feedback.missing_skills %}
              <li><strong>Core skills:</strong> {{ feedback.missing_skills|join(", ") }}</li>
            {% endif %}
            {% if feedback.gpa_advice %}
              <li><strong>GPA:</strong> {{ feedback.gpa_advice }}</li>
            {% endif %}
            {% if feedback.experience_advice %}
              <li><strong>Experience:</strong> {{ feedback.experience_advice }}</li>
            {% endif %}
            {% if feedback.project_advice %}
              <li><strong>Projects:</strong> {{ feedback.project_advice }}</li>
            {% endif %}
            {% if feedback.coursework_advice %}
              <li><strong>Coursework:</strong> {{ feedback.coursework_advice }}</li>
            {% endif %}
          </ul>
        </div>
      </div>

      {% if feedback.advice %}
      <div style="margin-top:14px;">
        <div style="font-weight:600; font-size:14px; margin-bottom:4px;">Detailed Suggestions</div>
        <div class="muted" style="font-size:12px; margin-bottom:6px;">
          These are grouped by theme (skills, target wording, title alignment, cleanup).
        </div>
        <div style="max-height:360px; overflow:auto; border-radius:8px; border:1px solid #111827; background:#020617; padding:10px;">
<pre style="white-space:pre-wrap; font-size:11px; margin:0;">{{ feedback.raw_json }}</pre>
        </div>
      </div>
      {% endif %}
    </div>
    {% endif %}

    {% endif %}

  </div>

  <script>
    // ---------- Log refresh (Parser tab) ----------
    function refreshLog(){
      fetch("{{ url_for('get_log') }}?path={{ log_file }}")
        .then(r => r.json())
        .then(data => {
          const ta = document.getElementById("logText");
          if(ta){ ta.value = data.text || ""; ta.scrollTop = ta.scrollHeight; }
        })
        .catch(err => console.error(err));
    }

    // ---------- Charts & clusters (Insights tab) ----------
    const chartsData = {{ charts | tojson }};
    const clustersData = {{ clusters | tojson }};

    function drawBarChart(canvasId, labels, values, label){
      const el = document.getElementById(canvasId);
      if(!el || !labels || labels.length === 0){ return; }
      const ctx = el.getContext('2d');
      new Chart(ctx, {
        type: 'bar',
        data: { labels: labels, datasets: [{ label: label, data: values }] },
        options: {
          responsive:true,
          plugins:{ legend:{ display:false } },
          scales:{
            x:{ ticks:{ maxRotation:60, minRotation:30 } },
            y:{ beginAtZero:true }
          }
        }
      });
    }

    // Only attempt to render on pages where canvases exist
    drawBarChart('skillsChart', chartsData.skills?.labels, chartsData.skills?.values, 'Count');
    drawBarChart('companiesChart', chartsData.companies?.labels, chartsData.companies?.values, 'Count');
    drawBarChart('titlesChart', chartsData.titles?.labels, chartsData.titles?.values, 'Count');
    drawBarChart('schoolsChart', chartsData.schools?.labels, chartsData.schools?.values, 'Count');
    drawBarChart('gpaChart', chartsData.gpa_dist?.labels, chartsData.gpa_dist?.values, 'Count');
    drawBarChart('skillsPerResumeChart', chartsData.skills_per_resume?.labels, chartsData.skills_per_resume?.values, 'Count');

    function renderClusters(){
      const root = document.getElementById('clusterSummary');
      if(!root){ return; }
      if(!clustersData || !clustersData.summaries || clustersData.summaries.length === 0){
        root.innerHTML = '<div class="muted">No clusters available.</div>';
        return;
      }
      const items = clustersData.summaries.map(c => {
        const terms = c.top_terms.map(t => `<span class="pill">${t}</span>`).join('');
        return `<div style="margin:8px 0; font-size:13px;">
          <div><strong>Cluster ${c.cluster}</strong> — ${c.size} resumes</div>
          <div style="margin-top:6px;">${terms}</div>
        </div>`;
      }).join('');
      root.innerHTML = items;
    }
    renderClusters();
  </script>
</body>
</html>
"""


# Helper to render any tab with shared context
def _render(active_tab: str,
            df: pd.DataFrame,
            charts: Dict[str, Any],
            clusters: Dict[str, Any],
            message: str = "",
            rows_to_show: int = 1,
            log_contents: str = "",
            feedback: Dict[str, Any] | None = None) -> str:
    charts, clusters = _safe_dashboard_payload(charts, clusters)  # guard payloads

    parsed_rows = int(len(df) if df is not None else 0)  # total rows
    max_table_rows = max(1, parsed_rows or 1)            # max rows for selector

    ctx = dict(
        active_tab=active_tab,
        folder_path=DEFAULT_FOLDER,
        output_csv=DEFAULT_OUTPUT_CSV,
        log_file=DEFAULT_LOG_FILE,
        limit=DEFAULT_LIMIT,
        max_workers=DEFAULT_WORKERS,
        parsed_rows=parsed_rows,
        table_rows=_table_rows(df, rows_to_show) if df is not None else [],
        charts=charts,
        clusters=clusters,
        message=message,
        rows_to_show=rows_to_show,
        max_table_rows=max_table_rows,
        log_contents=log_contents,
        feedback=feedback,
        model_ready=(MODEL.get("vectorizer") is not None and MODEL.get("kmeans") is not None),
    )
    return render_template_string(TEMPLATE, **ctx)  # render template


# Routes: Parser tab
@app.route("/", methods=["GET"])
@app.route("/parser", methods=["GET"])
def parser():
    # CSV and insights for table and counts
    df, charts, clusters = build_insights_from_csv(DEFAULT_OUTPUT_CSV)

    # rows-to-show selector (default 1, but bounded by CSV length)
    try:
        rows = int(request.args.get("rows", "1"))
    except Exception:
        rows = 1
    rows = max(1, min(rows, max(1, len(df) or 1)))

    # log contents (tail only recent lines)
    log_contents = _tail_log(DEFAULT_LOG_FILE, max_lines=400)

    return _render(
        active_tab="parser",
        df=df,
        charts=charts,
        clusters=clusters,
        message="",
        rows_to_show=rows,
        log_contents=log_contents,
        feedback=None,
    )


@app.route("/run", methods=["POST"])
def run_parse():
    """
    Run the batch parser (resume_batch_parser.batch_parse_resumes).

    We keep logging responsibility inside resume_batch_parser so tqdm
    still prints progress in your terminal. This route just triggers it
    and then reloads the parser tab with updated CSV + log contents.
    """
    folder_path = request.form.get("folder_path", DEFAULT_FOLDER)
    output_csv = request.form.get("output_csv", DEFAULT_OUTPUT_CSV)
    log_file = request.form.get("log_file", DEFAULT_LOG_FILE)

    try:
        limit = int(request.form.get("limit", DEFAULT_LIMIT))
    except Exception:
        limit = DEFAULT_LIMIT
    try:
        max_workers = int(request.form.get("max_workers", DEFAULT_WORKERS))
    except Exception:
        max_workers = DEFAULT_WORKERS

    # Run parser
    try:
        batch_parse_resumes(
            folder_path,
            output_csv,
            log_file,
            limit=limit,
            max_workers=max_workers,
        )
        message = f"Batch parse complete. Output saved to {output_csv}."
    except Exception as e:
        message = f"Error running batch parser: {e}"

    # Rebuild insights from new CSV
    df, charts, clusters = build_insights_from_csv(output_csv)

    # Read log contents after run (tail only)
    log_contents = _tail_log(log_file, max_lines=400)

    # Default to showing a single row unless user later changes it
    return _render(
        active_tab="parser",
        df=df,
        charts=charts,
        clusters=clusters,
        message=message,
        rows_to_show=1,
        log_contents=log_contents,
        feedback=None,
    )


@app.route("/logs", methods=["GET"])
def get_log():
    """
    Small JSON API so the log textarea can refresh without page reload.
    """
    path = request.args.get("path", DEFAULT_LOG_FILE)
    text = _tail_log(path, max_lines=400)  # tail recent lines
    return jsonify({"text": text})


# Routes: Training & Insights tab
@app.route("/insights", methods=["GET"])
def training_insights():
    df, charts, clusters = build_insights_from_csv(DEFAULT_OUTPUT_CSV)
    return _render(
        active_tab="insights",
        df=df,
        charts=charts,
        clusters=clusters,
        message="",
        rows_to_show=1,
        log_contents="",
        feedback=None,
    )


@app.route("/train", methods=["POST"])
def train():
    csv_path = request.form.get("csv_path", DEFAULT_OUTPUT_CSV)
    df, charts, clusters = build_insights_from_csv(csv_path)
    train_models_from_df(df)
    message = f"Models trained from {csv_path}."
    return _render(
        active_tab="insights",
        df=df,
        charts=charts,
        clusters=clusters,
        message=message,
        rows_to_show=1,
        log_contents="",
        feedback=None,
    )


@app.route("/reset_models", methods=["POST"])
def reset_models():
    """
    Clear the in-memory model and reset charts/clusters to defaults.
    """
    train_models_from_df(None)  # clear analyzer state

    # Wipe charts and cluster summaries in the UI
    empty_df = pd.DataFrame()
    empty_charts = {
        k: {"labels": [], "values": []}
        for k in DEFAULT_CHARTS.keys()
    }
    empty_clusters = {"summaries": []}

    message = "Analyzer models, charts, and cluster summaries have been reset."
    return _render(
        active_tab="insights",
        df=empty_df,
        charts=empty_charts,
        clusters=empty_clusters,
        message=message,
        rows_to_show=1,
        log_contents="",
        feedback=None,
    )


@app.route("/api/charts", methods=["GET"])
def api_charts():
    """
    JSON endpoint (optional) if you ever want to build a separate frontend.
    """
    csv_path = request.args.get("csv", DEFAULT_OUTPUT_CSV)
    _, charts, clusters = build_insights_from_csv(csv_path)
    charts, clusters = _safe_dashboard_payload(charts, clusters)
    return jsonify({"charts": charts, "clusters": clusters})


# Routes: Resume Analyzer tab
@app.route("/analyzer", methods=["GET", "POST"])
def resume_analyzer():
    """
    GET -> show analyzer form (tab 3)
    POST -> run analysis, show summary + recommendations
    """
    df, charts, clusters = build_insights_from_csv(DEFAULT_OUTPUT_CSV)

    if request.method == "GET":
        # Just render empty analyzer tab
        return _render(
            active_tab="analyzer",
            df=df,
            charts=charts,
            clusters=clusters,
            message="",
            rows_to_show=1,
            log_contents="",
            feedback=None,
        )

    # POST: user submitted a resume for analysis
    if MODEL.get("vectorizer") is None or MODEL.get("kmeans") is None:
        # If models are not trained, push user over to insights tab with hint
        msg = "No trained models yet. Go to the Training & Insights tab and click 'Train Models' first."
        return _render(
            active_tab="insights",
            df=df,
            charts=charts,
            clusters=clusters,
            message=msg,
            rows_to_show=1,
            log_contents="",
            feedback=None,
        )

    f = request.files.get("resume")           # uploaded file
    target = request.form.get("target", "")   # optional target keywords
    if not f:
        # Missing file -> just reload analyzer tab
        return redirect(url_for("resume_analyzer"))

    # Save upload
    original_name = f.filename or "resume"
    ext = os.path.splitext(original_name)[1].lower() or ".pdf"
    safe_name = slugify(original_name) + ext
    path = os.path.join(UPLOAD_DIR, safe_name)
    f.save(path)

    # Parse resume with your existing helper
    parsed = process_resume(path)
    single_df = pd.DataFrame([parsed]).fillna("")

    # Build document text mirroring training prep
    chunks: List[str] = []
    chunks.append(single_df.iloc[0].get("summary", ""))               # summary
    chunks.append(single_df.iloc[0].get("relevant_classwork", ""))    # coursework

    exp_list = _ensure_list(single_df.iloc[0].get("experience", ""))  # experience
    for item in exp_list:
        item = item or {}
        chunks.append(_s(item.get("title", "")))                      # title
        chunks.append(_s(item.get("company", "")))                    # company
        bullets = item.get("bullets", []) or []
        if isinstance(bullets, list):
            chunks.extend([_s(b) for b in bullets])                   # add bullets

    proj_list = _ensure_list(single_df.iloc[0].get("projects", ""))  # projects
    for item in proj_list:
        item = item or {}
        chunks.append(_s(item.get("name", "")))                       # project name
        chunks.append(_s(item.get("description", "")))                # project desc

    res_list = _ensure_list(single_df.iloc[0].get("research", ""))   # research
    for item in res_list:
        item = item or {}
        chunks.append(_s(item.get("title", "")))                      # research title
        chunks.append(_s(item.get("description", "")))                # research desc

    edu_list = _ensure_list(single_df.iloc[0].get("education", ""))  # education
    for item in edu_list:
        item = item or {}
        chunks.append(_s(item.get("school", "")))                     # school
        chunks.append(_s(item.get("degree", "")))                     # degree

    text = "\n".join([c for c in chunks if isinstance(c, str)])      # final doc text

    # Predict cluster
    vec = MODEL["vectorizer"]
    kmeans = MODEL["kmeans"]
    x = vec.transform([text])
    cl = int(kmeans.predict(x)[0])

    # Optional role alignment score (cosine similarity between cluster centroid and target text)
    role_score = None
    if target:
        import numpy as np  # local import for cosine
        role_vec = vec.transform([target])
        centroid = kmeans.cluster_centers_[cl].reshape(1, -1)
        num = float((role_vec @ centroid.T)[0, 0])             # dot product
        denom = (np.linalg.norm(role_vec.toarray()) * np.linalg.norm(centroid)) or 1.0
        role_score = num / denom

    # Skill gaps
    user_skills = {s.strip() for s in (single_df.iloc[0].get("skills", "") or "").split(",") if s.strip()}
    user_skills_norm = {_norm_skill_token(s) for s in user_skills}
    cluster_skills = MODEL.get("cluster_skill_vocab", {}).get(cl, {})
    top_cluster_skills = [k for k, _ in sorted(cluster_skills.items(), key=lambda kv: -kv[1])[:20]]
    missing_core = [s for s in top_cluster_skills if s not in user_skills_norm][:10]
    redundancy = [s for s in user_skills_norm if s and s not in MODEL.get("skill_vocab", {})][:10]

    likely_titles = MODEL.get("title_labels", [])[cl] if MODEL.get("title_labels") else []

    # GPA extraction (safe against strings)
    gpa_value = parsed.get("gpa")
    if not gpa_value and edu_list:
        first = edu_list[0] or {}
        gpa_value = first.get("gpa") or first.get("GPA")

    gpa_numeric = None
    gpa_display = None
    if gpa_value:
        gpa_display = str(gpa_value)
        try:
            gpa_numeric = float(str(gpa_value).split()[0])
        except Exception:
            gpa_numeric = None

    # Counselor-style advice structure (kept for raw JSON view)
    advice = []
    if missing_core:
        advice.append({"type": "skills",
                       "text": "Add or demonstrate these core skills for this cluster",
                       "items": missing_core})
    if target:
        advice.append({"type": "target",
                       "text": "Align wording towards your target role keywords",
                       "items": [target]})
    if likely_titles:
        advice.append({"type": "titles",
                       "text": "Roles similar to your profile in our corpus",
                       "items": likely_titles})
    if redundancy:
        advice.append({"type": "cleanup",
                       "text": "These skills look uncommon/noisy—merge synonyms or show outcomes",
                       "items": list(redundancy)})

    # Higher-level heuristic advice (experience, projects, GPA, coursework)
    section_counts = {
        "experience": len(exp_list),
        "projects": len(proj_list),
        "research": len(res_list),
        "education": len(edu_list),
    }

    # Experience advice
    if section_counts["experience"] == 0:
        experience_advice = "No experience entries detected. Add internships, part-time roles, or significant volunteer work."
    elif section_counts["experience"] == 1:
        experience_advice = "Only one experience entry detected. Consider adding more detail or additional roles if you have them."
    else:
        experience_advice = ""

    # Projects advice
    if section_counts["projects"] == 0:
        project_advice = "No projects detected. Add 2–4 concrete projects that demonstrate your skills."
    elif section_counts["projects"] == 1:
        project_advice = "Only one project detected. Add another project or expand the existing one with more impact-focused bullets."
    else:
        project_advice = ""

    # Coursework advice
    summary_text = _s(parsed.get("summary", ""))
    classwork_text = _s(parsed.get("relevant_classwork", ""))
    if not classwork_text and section_counts["experience"] < 2:
        coursework_advice = "Relevant coursework is empty. For internships or early career roles, add 4–6 targeted courses."
    else:
        coursework_advice = ""

    # GPA advice
    if gpa_numeric is None and not gpa_display:
        gpa_advice = "No GPA detected. If your GPA is strong (e.g., 3.0+), consider including it for internships and early career roles."
    elif gpa_numeric is not None and gpa_numeric < 3.0:
        gpa_advice = "GPA appears below 3.0. Focus your resume on strong projects and experience; only include GPA when required."
    else:
        gpa_advice = ""

    feedback = {
        "cluster": cl,
        "role_score": role_score,
        "likely_titles": likely_titles,
        "missing_skills": missing_core,
        "advice": advice,
        "parsed_contact": {
            "name": parsed.get("name") or (parsed.get("contact", {}) or {}).get("name", ""),
            "email": parsed.get("email") or (parsed.get("contact", {}) or {}).get("email", ""),
        },
        "gpa_display": gpa_display,
        "gpa_advice": gpa_advice,
        "experience_advice": experience_advice,
        "project_advice": project_advice,
        "coursework_advice": coursework_advice,
        "section_counts": section_counts,
        # raw JSON string for the detail panel
        "raw_json": json.dumps({
            "cluster": cl,
            "role_score": role_score,
            "likely_titles": likely_titles,
            "missing_skills": missing_core,
            "advice": advice,
            "parsed_contact": {
                "name": parsed.get("name") or (parsed.get("contact", {}) or {}).get("name", ""),
                "email": parsed.get("email") or (parsed.get("contact", {}) or {}).get("email", ""),
            },
        }, indent=2, ensure_ascii=False),
    }

    msg = (
        f"Analyzed upload. Cluster = {cl}. "
        + (f"Role match score = {role_score:.3f}. " if role_score is not None else "")
        + (
            f"Suggested roles: {', '.join(likely_titles[:3])}. "
            if likely_titles else "Suggested roles: N/A. "
        )
        + (
            f"Missing core skills: {', '.join(missing_core[:5])}."
            if missing_core else ""
        )
    )

    return _render(
        active_tab="analyzer",
        df=df,
        charts=charts,
        clusters=clusters,
        message=msg,
        rows_to_show=1,
        log_contents="",
        feedback=feedback,
    )


# Entrypoint for local dev
if __name__ == "__main__":           # script run guard
    app.run(debug=True)              # start Flask dev server
