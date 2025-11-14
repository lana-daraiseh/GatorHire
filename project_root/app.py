# Future annotations for forward references
from __future__ import annotations  # enable postponed evaluation of annotations

# Stdlib imports
import json  # JSON utilities
import os    # filesystem utilities
import math  # math helpers
from typing import Any, Dict, List, Tuple  # type hints

# Third-party and framework imports
from flask import Flask, render_template_string, request, redirect, url_for, jsonify  # Flask tools
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
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024   # 25MB upload limit
UPLOAD_DIR = "uploads"                                # uploads directory
os.makedirs(UPLOAD_DIR, exist_ok=True)                # ensure uploads dir exists

# In-memory model store for quick inference
MODEL = {
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
DEFAULT_CHARTS = {"skills": {"labels": [], "values": []},
                  "companies": {"labels": [], "values": []}}  # default charts
DEFAULT_CLUSTERS = {"summaries": []}                           # default cluster summaries

# Guard dashboard payload shape to avoid template or JS errors
def _safe_dashboard_payload(charts, clusters):
    if not isinstance(charts, dict) or "skills" not in charts or "companies" not in charts:
        charts = DEFAULT_CHARTS  # fallback charts
    if not isinstance(clusters, dict) or "summaries" not in clusters:
        clusters = DEFAULT_CLUSTERS  # fallback clusters
    return charts, clusters  # always safe structures

# String sanitization helper
def _s(x: Any) -> str:
    return x.strip() if isinstance(x, str) else ""  # trimmed str or empty

# Dashboard HTML template (single-file app UI)
# Includes forms for batch parsing, training, and single-resume analysis; Chart.js for charts; cluster summaries; table view.
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
  :root{ --bg:#0b0f14; --card:#111827; --muted:#9ca3af; --txt:#e5e7eb; --accent:#60a5fa; }
  *{ box-sizing:border-box; }
  body{ margin:0; font-family:Inter, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial; background:var(--bg); color:var(--txt); }
  .wrap{ max-width:1200px; margin:0 auto; padding:24px; }
  .h1{ font-size:28px; font-weight:800; letter-spacing:0.2px; }
  .muted{ color:var(--muted); }
  .grid{ display:grid; gap:16px; }
  .grid-2{ grid-template-columns: 1fr 1fr; }
  .card{ background:var(--card); border:1px solid #1f2937; border-radius:16px; padding:16px; box-shadow:0 10px 30px rgba(0,0,0,.25); }
  .btn{ background:var(--accent); color:#0b0f14; padding:10px 14px; border:none; border-radius:10px; font-weight:600; cursor:pointer; }
  .btn:disabled{ opacity:0.6; cursor:not-allowed; }
  input, select{ width:100%; background:#0f172a; color:var(--txt); border:1px solid #1f2937; padding:10px 12px; border-radius:10px; }
  label{ font-weight:600; font-size:12px; color:#9ca3af; text-transform:uppercase; letter-spacing:0.06em; }
  table{ width:100%; border-collapse:collapse; font-size:14px; table-layout:fixed; }
  th, td{ border-bottom:1px solid #1f2937; padding:10px 8px; text-align:left; vertical-align:middle; word-wrap:break-word; }
  th{ color:#9ca3af; font-weight:600; }
  tr{ height:1px; }
  code{ background:#0f172a; color:#a7f3d0; padding:2px 6px; border-radius:6px; }
  .pill{ display:inline-block; padding:3px 8px; background:#0f172a; border:1px solid #1f2937; border-radius:999px; margin:2px 4px 2px 0; font-size:12px; line-height:1.3em; white-space:nowrap; }
</style>
</head>
<body>
  <div class="wrap">
    <div class="grid" style="gap:8px; margin-bottom:8px;">
      <div class="h1">Resume Parser Dashboard</div>
    </div>

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
        {% if parsed_rows > 0 %}
        <a class="btn" style="background:#34d399;" href="{{ url_for('refresh_data') }}">Refresh Insights</a>
        {% endif %}
      </div>
    </form>

    <form class="card" method="post" action="{{ url_for('train') }}" style="margin-top:12px;">
      <div class="grid grid-2">
        <div>
          <label>Train from CSV</label>
          <input name="csv_path" value="{{ output_csv }}" />
        </div>
      </div>
      <div style="margin-top:12px; display:flex; gap:8px;">
        <button class="btn" type="submit">Train Models</button>
      </div>
    </form>

    <form class="card" method="post" action="{{ url_for('analyze_upload') }}" enctype="multipart/form-data" style="margin-top:12px;">
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
        <button class="btn" type="submit">Analyze Resume & Get Advice</button>
      </div>
      <div class="muted" style="margin-top:8px;">We compare your resume to learned clusters and highlight missing skills, keywords, and action items.</div>
    </form>

    {% if message %}
      <div class="card">
            <pre>{{ message | safe }}</pre>
      </div>
    {% endif %}

    {% if parsed_rows > 0 %}
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

    <div class="grid" style="margin-top:16px;">
      <div class="card">
        <h3 style="margin:4px 0 12px;">Clusters (KMeans)</h3>
        <div class="muted" style="margin-bottom:8px;">Each resume is labeled by a cluster; top keywords per cluster summarize themes.</div>
        <div id="clusterSummary"></div>
      </div>
    </div>

    <div class="card" style="margin-top:16px;">
      <h3 style="margin:4px 0 12px;">Parsed Resumes</h3>
      <div class="muted" style="margin-bottom:8px;">Showing up to first 20 rows.</div>
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
        </tbody>
      </table>
    </div>
    {% endif %}
  </div>

  <script>
    const skillsData = {{ charts.skills | tojson }};
    const companiesData = {{ charts.companies | tojson }};
    const clusterData = {{ clusters | tojson }};

    function drawBarChart(canvasId, labels, values, label){
      const ctx = document.getElementById(canvasId).getContext('2d');
      new Chart(ctx, {
        type: 'bar',
        data: { labels: labels, datasets: [{ label: label, data: values }] },
        options: { responsive:true, plugins:{ legend:{ display:false } } }
      });
    }

    if(skillsData.labels){ drawBarChart('skillsChart', skillsData.labels, skillsData.values, 'Count'); }
    if(companiesData.labels){ drawBarChart('companiesChart', companiesData.labels, companiesData.values, 'Count'); }

    function renderClusters(){
      const root = document.getElementById('clusterSummary');
      if(!clusterData || !clusterData.summaries){ root.innerHTML = '<div class="muted">No clusters available.</div>'; return; }
      const items = clusterData.summaries.map(c => {
        const terms = c.top_terms.map(t => `<span class="pill">${t}</span>`).join('');
        return `<div style="margin:8px 0;">
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

# JSON coercion helper
def _parse_json_field(x: Any) -> Any:
    if isinstance(x, (list, dict)):
        return x  # already structured
    try:
        return json.loads(x) if isinstance(x, str) and x.strip().startsWith(('{', '[')) else x  # parse JSON-like text
    except Exception:
        return [] if isinstance(x, str) else x  # string parse failure -> []

# Skill token normalization for counting and matching
def _norm_skill_token(s: str) -> str:
    s = (s or "").strip()  # trim
    if not s:
        return ""  # empty guard
    s = s.replace("/", " /")  # separate slashed items
    return " ".join(s.split()).lower()  # collapse spaces and lowercase

# Build insights (df, charts, clusters) from CSV on disk
def build_insights_from_csv(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
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

    # Company aggregation from experience
    companies: List[str] = []  # company names
    if "experience" in df.columns:
        for cell in df["experience"].tolist():
            exp_list = _parse_json_field(cell) or []  # parse JSON cell
            if isinstance(exp_list, list):
                for item in exp_list:
                    comp = _s((item or {}).get("company", ""))  # company field
                    if comp:
                        companies.append(comp)  # collect

    company_freq: Dict[str, int] = {}  # frequency map
    for c in companies:
        company_freq[c] = company_freq.get(c, 0) + 1  # count

    top_companies = sorted(company_freq.items(), key=lambda kv: (-kv[1], slugify(kv[0] or "")))[:20]  # top 20
    companies_chart = {"labels": [k for k, _ in top_companies], "values": [v for _, v in top_companies]}  # chart payload

    # Build text corpus for clustering
    texts: List[str] = []     # documents
    index_map: List[int] = [] # doc->df index mapping

    for ix, row in df.iterrows():
        chunks: List[str] = []  # textual pieces

        # main fields
        chunks.append(row.get("summary", ""))               # summary text
        chunks.append(row.get("relevant_classwork", ""))    # course text

        # experience fields
        exp_list = _parse_json_field(row.get("experience", "")) or []
        if isinstance(exp_list, list):
            for item in exp_list:
                chunks.append(_s((item or {}).get("title", "")))    # job title
                chunks.append(_s((item or {}).get("company", "")))  # company
                bullets = (item or {}).get("bullets", []) or []     # bullet list
                if isinstance(bullets, list):
                    chunks.extend([_s(b) for b in bullets])          # add bullets

        # projects fields
        proj_list = _parse_json_field(row.get("projects", "")) or []
        if isinstance(proj_list, list):
            for item in proj_list:
                chunks.append(_s((item or {}).get("name", "")))        # project name
                chunks.append(_s((item or {}).get("description", "")))  # project desc

        # research fields
        res_list = _parse_json_field(row.get("research", "")) or []
        if isinstance(res_list, list):
            for item in res_list:
                chunks.append(_s((item or {}).get("title", "")))        # research title
                chunks.append(_s((item or {}).get("description", "")))   # research desc

        # education fields
        edu_list = _parse_json_field(row.get("education", "")) or []
        if isinstance(edu_list, list):
            for item in edu_list:
                chunks.append(_s((item or {}).get("school", "")))       # school
                chunks.append(_s((item or {}).get("degree", "")))       # degree

        text = "\n".join([c for c in chunks if isinstance(c, str)])  # join to doc
        if isinstance(text, str) and len(text.split()) > 2:          # min length
            texts.append(text)                                        # keep
            index_map.append(ix)                                      # map

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
            summaries.append({"cluster": int(i), "size": size, "top_terms": top_terms})  # summary
        clusters_dict = {"summaries": summaries}                         # final dict

    charts = {"skills": skills_chart, "companies": companies_chart}  # charts bundle
    return df, charts, clusters_dict  # return insights

# Train TF-IDF + KMeans, build skill/title vocabularies, store in MODEL
def train_models_from_df(df: pd.DataFrame):
    global MODEL  # mutate module-level cache

    if df is None or df.empty:
        MODEL.update({
            "vectorizer": None, "kmeans": None, "terms": None, "labels": None,
            "df_index_map": None, "skill_vocab": {}, "cluster_skill_vocab": {}, "title_labels": [],
        })  # clear state
        return

    texts = []      # training docs
    index_map = []  # doc->df row
    for ix, row in df.iterrows():
        chunks = []                                                  # gather fields
        chunks.append(row.get("summary", ""))                        # summary
        chunks.append(row.get("relevant_classwork", ""))             # coursework
        exp_list = _parse_json_field(row.get("experience", "")) or []  # experience
        if isinstance(exp_list, list):
            for item in exp_list:
                chunks.append(_s((item or {}).get("title", "")))     # title
                chunks.append(_s((item or {}).get("company", "")))   # company
                bullets = (item or {}).get("bullets", []) or []      # bullets
                if isinstance(bullets, list):
                    chunks.extend([_s(b) for b in bullets])           # add bullets
        proj_list = _parse_json_field(row.get("projects", "")) or []   # projects
        if isinstance(proj_list, list):
            for item in proj_list:
                chunks.append(_s((item or {}).get("name", "")))        # project name
                chunks.append(_s((item or {}).get("description", "")))  # project desc
        res_list = _parse_json_field(row.get("research", "")) or []     # research
        if isinstance(res_list, list):
            for item in res_list:
                chunks.append(_s((item or {}).get("title", "")))        # research title
                chunks.append(_s((item or {}).get("description", "")))   # research desc
        edu_list = _parse_json_field(row.get("education", "")) or []     # education
        if isinstance(edu_list, list):
            for item in edu_list:
                chunks.append(_s((item or {}).get("school", "")))       # school
                chunks.append(_s((item or {}).get("degree", "")))       # degree
        text = "\n".join([c for c in chunks if isinstance(c, str)])     # document
        if isinstance(text, str) and len(text.split()) > 2:             # min length
            texts.append(text)                                          # keep
            index_map.append(ix)                                        # map

    if not texts:
        MODEL.update({
            "vectorizer": None, "kmeans": None, "terms": None, "labels": None,
            "df_index_map": None, "skill_vocab": {}, "cluster_skill_vocab": {}, "title_labels": [],
        })  # clear if no data
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
    skill_freq = {}                               # global freq
    cluster_skill = {i: {} for i in range(k)}     # per-cluster freq

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
    title_labels = [[] for _ in range(k)]  # titles per cluster
    if "experience" in df.columns:
        for local_i, df_i in enumerate(index_map):
            cid = int(labels[local_i])                                    # cluster id
            exp_list = _parse_json_field(df.iloc[df_i].get("experience", "")) or []  # exp list
            if isinstance(exp_list, list):
                for item in exp_list:
                    t = _s((item or {}).get("title", ""))                 # title
                    if t:
                        title_labels[cid].append(t)                        # collect

    MODEL["title_labels"] = [
        [t for t, _ in sorted({t: title_labels[i].count(t) for t in title_labels[i]}.items(), key=lambda kv: -kv[1])[:5]]
        for i in range(k)
    ]  # top-5 titles per cluster

# Route: dashboard index
@app.route("/", methods=["GET"])
def index():
    df, charts, clusters = build_insights_from_csv(DEFAULT_OUTPUT_CSV)  # build from default CSV
    charts, clusters = _safe_dashboard_payload(charts, clusters)        # guard payload
    table_rows = _table_rows(df)                                        # compact table rows
    return render_template_string(                                      # render page
        TEMPLATE,
        folder_path=DEFAULT_FOLDER,
        output_csv=DEFAULT_OUTPUT_CSV,
        log_file=DEFAULT_LOG_FILE,
        limit=DEFAULT_LIMIT,
        max_workers=DEFAULT_WORKERS,
        message="",
        parsed_rows=len(df),
        table_rows=table_rows,
        charts=charts,
        clusters=clusters,
    )

# Route: run batch parser and retrain models
@app.route("/run", methods=["POST"])
def run_parse():
    folder_path = request.form.get("folder_path", DEFAULT_FOLDER)       # input folder
    output_csv = request.form.get("output_csv", DEFAULT_OUTPUT_CSV)     # output CSV path
    log_file = request.form.get("log_file", DEFAULT_LOG_FILE)           # log path
    try:
        limit = int(request.form.get("limit", DEFAULT_LIMIT))           # parse limit
    except Exception:
        limit = DEFAULT_LIMIT                                           # fallback
    try:
        max_workers = int(request.form.get("max_workers", DEFAULT_WORKERS))  # worker count
    except Exception:
        max_workers = DEFAULT_WORKERS                                   # fallback

    try:
        batch_parse_resumes(folder_path, output_csv, log_file, limit=limit, max_workers=max_workers)  # run batch parse
        message = f"Batch parse complete. Output saved to {output_csv}."                               # success msg
    except Exception as e:
        message = f"Error running batch parser: {e}"                                                    # error msg

    df, charts, clusters = build_insights_from_csv(output_csv)  # rebuild insights from new CSV
    train_models_from_df(df)                                   # retrain models
    charts, clusters = _safe_dashboard_payload(charts, clusters)  # safe payloads
    table_rows = _table_rows(df)                                   # table rows

    return render_template_string(  # render page
        TEMPLATE,
        folder_path=folder_path,
        output_csv=output_csv,
        log_file=log_file,
        limit=limit,
        max_workers=max_workers,
        message=message + " Models trained.",
        parsed_rows=len(df),
        table_rows=table_rows,
        charts=charts,
        clusters=clusters,
    )

# Route: refresh visuals and models from existing CSV (no parsing)
@app.route("/refresh", methods=["GET"])
def refresh_data():
    csv_path = request.args.get("csv", DEFAULT_OUTPUT_CSV)        # CSV override
    df, charts, clusters = build_insights_from_csv(csv_path)      # build insights
    train_models_from_df(df)                                      # retrain models
    charts, clusters = _safe_dashboard_payload(charts, clusters)  # guard payload
    table_rows = _table_rows(df)                                  # table rows
    return render_template_string(                                # render page
        TEMPLATE,
        folder_path=DEFAULT_FOLDER,
        output_csv=csv_path,
        log_file=DEFAULT_LOG_FILE,
        limit=DEFAULT_LIMIT,
        max_workers=DEFAULT_WORKERS,
        message="Refreshed from CSV and models trained.",
        parsed_rows=len(df),
        table_rows=table_rows,
        charts=charts,
        clusters=clusters,
    )

# Route: train models from chosen CSV (no parsing)
@app.route("/train", methods=["POST"])
def train():
    csv_path = request.form.get("csv_path", DEFAULT_OUTPUT_CSV)   # CSV path
    df, charts, clusters = build_insights_from_csv(csv_path)      # build insights
    train_models_from_df(df)                                      # train models
    charts, clusters = _safe_dashboard_payload(charts, clusters)  # guard payload
    table_rows = _table_rows(df)                                  # table rows
    return render_template_string(                                # render page
        TEMPLATE,
        folder_path=DEFAULT_FOLDER,
        output_csv=csv_path,
        log_file=DEFAULT_LOG_FILE,
        limit=DEFAULT_LIMIT,
        max_workers=DEFAULT_WORKERS,
        message=f"Models trained from {csv_path}.",
        parsed_rows=len(df),
        table_rows=table_rows,
        charts=charts,
        clusters=clusters,
    )

# Route: analyze a single uploaded resume against trained clusters
@app.route("/analyze", methods=["POST"])
def analyze_upload():
    if MODEL.get("vectorizer") is None:                                  # need model
        df_try, charts_try, clusters_try = build_insights_from_csv(DEFAULT_OUTPUT_CSV)  # try bootstrap
        if not df_try.empty:
            train_models_from_df(df_try)                                 # train if data exists
        else:
            charts_try, clusters_try = _safe_dashboard_payload(charts_try, clusters_try)  # safe payload
            return render_template_string(                               # prompt to train
                TEMPLATE,
                folder_path=DEFAULT_FOLDER,
                output_csv=DEFAULT_OUTPUT_CSV,
                log_file=DEFAULT_LOG_FILE,
                limit=DEFAULT_LIMIT,
                max_workers=DEFAULT_WORKERS,
                message="No trained models yet. Parse data and click Train first.",
                parsed_rows=0,
                table_rows=[],
                charts=charts_try,
                clusters=clusters_try,
            )

    f = request.files.get("resume")           # uploaded file
    target = request.form.get("target", "")   # optional target keywords
    if not f:
        return redirect(url_for('index'))     # missing file -> index

    safe_name = slugify(f.filename or "resume") + (os.path.splitext(f.filename or "resume.pdf")[1].lower())  # safe name
    path = os.path.join(UPLOAD_DIR, safe_name)  # upload path
    f.save(path)                                # save file

    parsed = process_resume(path)                         # parse file
    single_df = pd.DataFrame([parsed]).fillna("")         # single-row df

    # Build document text mirroring training prep
    chunks = []
    chunks.append(single_df.iloc[0].get("summary", ""))               # summary
    chunks.append(single_df.iloc[0].get("relevant_classwork", ""))    # coursework
    exp = _parse_json_field(single_df.iloc[0].get("experience", "")) or []  # experience
    if isinstance(exp, list):
        for item in exp:
            chunks.append(_s((item or {}).get("title", "")))          # title
            chunks.append(_s((item or {}).get("company", "")))        # company
            bullets = (item or {}).get("bullets", []) or []           # bullets
            if isinstance(bullets, list):
                chunks.extend([_s(b) for b in bullets])               # add bullets
    proj = _parse_json_field(single_df.iloc[0].get("projects", "")) or []    # projects
    if isinstance(proj, list):
        for item in proj:
            chunks.append(_s((item or {}).get("name", "")))           # project name
            chunks.append(_s((item or {}).get("description", "")))    # project desc
    res = _parse_json_field(single_df.iloc[0].get("research", "")) or []     # research
    if isinstance(res, list):
        for item in res:
            chunks.append(_s((item or {}).get("title", "")))          # research title
            chunks.append(_s((item or {}).get("description", "")))    # research desc
    edu = _parse_json_field(single_df.iloc[0].get("education", "")) or []    # education
    if isinstance(edu, list):
        for item in edu:
            chunks.append(_s((item or {}).get("school", "")))         # school
            chunks.append(_s((item or {}).get("degree", "")))         # degree

    text = "\n".join([c for c in chunks if isinstance(c, str)])  # final doc text

    vec = MODEL["vectorizer"]                 # vectorizer
    kmeans = MODEL["kmeans"]                  # kmeans model
    x = vec.transform([text])                 # vectorize
    cl = int(kmeans.predict(x)[0])            # predicted cluster

    role_score = None                         # optional role alignment
    if target:
        import numpy as np                    # local import for cosine
        role_vec = vec.transform([target])    # vectorize target text
        centroid = kmeans.cluster_centers_[cl].reshape(1, -1)  # cluster centroid
        num = float((role_vec @ centroid.T)[0, 0])             # dot product
        denom = (np.linalg.norm(role_vec.toarray()) * np.linalg.norm(centroid)) or 1.0  # cosine denom
        role_score = round(num / denom, 4)    # cosine similarity

    user_skills = set([s.strip() for s in (single_df.iloc[0].get("skills", "").split(",")) if s.strip()])  # user skills
    user_skills_norm = {_norm_skill_token(s) for s in user_skills}                                         # normalized
    cluster_skills = MODEL.get("cluster_skill_vocab", {}).get(cl, {})                                      # cluster vocab
    top_cluster_skills = [k for k, _ in sorted(cluster_skills.items(), key=lambda kv: -kv[1])[:20]]        # top 20
    missing_core = [s for s in top_cluster_skills if s not in user_skills_norm][:10]                       # gaps
    redundancy = [s for s in user_skills_norm if s and s not in MODEL.get("skill_vocab", {})][:10]         # uncommon

    likely_titles = MODEL.get("title_labels", [])[cl] if MODEL.get("title_labels") else []  # suggested titles

    advice = []  # guidance list
    if missing_core:
        advice.append({"type":"skills","text":"Add or demonstrate these core skills for this cluster","items": missing_core})  # skill gaps
    if target:
        advice.append({"type":"target","text":"Align wording towards your target role keywords","items": [target]})            # target align
    if likely_titles:
        advice.append({"type":"titles","text":"Roles similar to your profile in our corpus","items": likely_titles})           # titles
    if redundancy:
        advice.append({"type":"cleanup","text":"These skills look uncommon/noisy—merge synonyms or show outcomes","items": redundancy})  # cleanup

    df_all, charts, clusters = build_insights_from_csv(DEFAULT_OUTPUT_CSV)  # refresh visuals
    charts, clusters = _safe_dashboard_payload(charts, clusters)            # guard payload
    table_rows = _table_rows(df_all)                                        # table rows

    feedback = {                                   # structured feedback to display
        "cluster": cl,
        "role_score": role_score,
        "likely_titles": likely_titles,
        "missing_skills": missing_core,
        "advice": advice,
        "parsed_contact": {
            "name": parsed.get("name") or (parsed.get("contact", {}) or {}).get("name", ""),
            "email": parsed.get("email") or (parsed.get("contact", {}) or {}).get("email", ""),
        }
    }
    msg = "\n".join([
        f"📊 Cluster: {cl}",
        f"💡 Advice: ",
        (
                "⚠️ Missing core skills:\n" +
                "\n".join(f"   • {s}" for s in missing_core[:5])
        ) if missing_core else "",
        f"🎯 Role match score: {role_score}" if role_score is not None else "",
        f"💼 Suggested roles: {', '.join(likely_titles[:3]) if likely_titles else 'N/A'}",
        f"🏷️ Likely titles: {likely_titles}",
        f"📇 Parsed contact: {parsed.get('contact', {}).get('name', '')}",
    ]).strip()

    #msg = (f"Analyzed upload. Cluster = {cl}. \n" +                       # summary message
           #(f"Role match score = {role_score}. " if role_score is not None else "") +
          # f"Suggested roles: {', '.join(likely_titles[:3]) if likely_titles else 'N/A'}. \n" +
           #(f"Missing core skills: {', '.join(missing_core[:5])}." if missing_core else ""))

    return render_template_string(  # render dashboard with feedback blob
        TEMPLATE.replace("<div class=\"card\">{{ message }}</div>",
                          "<div class=\"card\">{{ message }}<pre style=\"white-space:pre-wrap; margin-top:8px; background:#0f172a; padding:8px; border-radius:8px;\">{{ feedback | tojson(indent=2) }}</pre></div>"),
        folder_path=DEFAULT_FOLDER,
        output_csv=DEFAULT_OUTPUT_CSV,
        log_file=DEFAULT_LOG_FILE,
        limit=DEFAULT_LIMIT,
        max_workers=DEFAULT_WORKERS,
        message=msg,
        parsed_rows=len(df_all),
        table_rows=table_rows,
        charts=charts,
        clusters=clusters,
        feedback=feedback,
    )

# Route: API for charts and clusters JSON
@app.route("/api/charts", methods=["GET"])
def api_charts():
    csv_path = request.args.get("csv", DEFAULT_OUTPUT_CSV)  # CSV override
    _, charts, clusters = build_insights_from_csv(csv_path) # build insights
    charts, clusters = _safe_dashboard_payload(charts, clusters)  # guard payload
    return jsonify({"charts": charts, "clusters": clusters})       # JSON response

# Compact education cell for table view (first 3 entries)
def _compact_education(cell: str) -> str:
    items = _parse_json_field(cell) or []     # parse cell JSON/list
    out: List[str] = []                       # pieces
    if isinstance(items, list):
        for it in items[:3]:
            school = _s((it or {}).get("school", ""))  # school
            degree = _s((it or {}).get("degree", ""))  # degree
            year = _s((it or {}).get("year", ""))      # year
            piece = ", ".join([p for p in [school, degree, year] if p])  # joined
            if piece:
                out.append(piece)            # add piece
    return " • ".join(out)                   # bullet join

# Turn a comma skills string into a deduped, capped list
def _skills_list(cell: str) -> List[str]:
    toks = [t.strip() for t in (cell or "").split(",") if t.strip()]  # split
    seen = set()                                                       # dedupe set
    out = []                                                           # output list
    for t in toks:
        low = t.lower()                                                # casefold
        if low not in seen:
            seen.add(low)                                              # mark
            out.append(t)                                              # keep original case
    return out[:12]                                                    # cap list

# Build up to 20 table rows for dashboard
def _table_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []            # output rows
    if df is None or df.empty:
        return rows                            # empty guard
    for _, r in df.head(20).iterrows():        # first 20 rows
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

# Entrypoint for local dev
if __name__ == "__main__":           # script run guard
    app.run(debug=True)              # start Flask dev server