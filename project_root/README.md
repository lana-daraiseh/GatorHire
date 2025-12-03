# Resume Insight Dashboard

A Flask-based dashboard that parses and analyzes resumes using Ollama and machine learning. It extracts structured data from PDF or DOCX resumes, stores results in a CSV file, and builds insights with TF-IDF and K-Means clustering.

## Features
- Batch resume parsing using Ollama (Gemma models)
- CSV export of parsed results
- Flask web interface for visualization
- TF-IDF + K-Means clustering for insights
- Refresh Insights button to rebuild charts without re-parsing
- Logging, concurrency, and error handling

## Project Structure
project_root/
│
├── app_v2.py                   # Main Flask app
├── Resumes/		     # Folder containing Resumes to be parsed
├── resume_batch_parser.py   # Handles batch parsing and CSV export
├── textExtractor.py            # Extracts text from PDF or DOCX files
├── parsed_resumes.csv       # Output CSV (created after parsing)
├── parser_log.log           # Log file
├── env/                     # Virtual environment
├── uploads                  # Uploaded Resumes (created after uploading)
└── README.md                # Documentation

## Environment Setup

1. Create and activate a virtual environment:
   .env\Scripts\activate        # Windows
   

2. Install required Python packages:
   pip install flask pandas scikit-learn python-slugify pdfplumber docx2txt tqdm PyPDF2

3. Install and verify Ollama (Can skip 3-5 if using API key):
   - Download from: https://ollama.com/download
   - Open a new PowerShell window and verify:
     ollama --version
   - If not recognized, add the path:
     C:\Users\<YourName>\AppData\Local\Programs\Ollama
     to your system environment variables.

4. Start the Ollama server:
   ollama serve

5. Pull and test a model:
   ollama pull gemma3:12
   ollama run gemma3:12 "hello"

   If you see a response, the model is ready. If memory errors occur, use a smaller model like gemma2:2b.

6. Get API Key:
   - Go to https://ai.ufl.edu/teaching-with-ai/for-uf-faculty/navigator/
   - Scroll down till you see "Get Started With NaviGator"
   - Click on Navigator Toolkit
   - In the tab Virtual Keys, click "Create New Key"
   - For team choose: navigator-toolkit
   - Name the key
   - For models choose: gemma-3-27b-it
   - Create Key

7. Add Key to resume_batch_parser.py:
   - Copy key and add it to api_key="" near the top of the file

8. Changing parsing
   - The API call is default, if you want to run a model locally follow:
        - In app_v2, switch resume_batch_parser with parser2 in this line:
        - from resume_batch_parser import batch_parse_resumes, process_resume

## Running the Application

1. Ensure the Ollama server is running (ollama serve).
2. Activate your virtual environment.
3. Run:
   python app.py
4. Open your browser to:
   http://127.0.0.1:5000

## Application Workflow

### 1. Parse Resumes
- Place resumes in the Resumes/ folder.
- Click "Run Batch Parse" in the web app.
- Each file is processed through the parser (tempParser.py) and then analyzed by Ollama.
- Results are saved into parsed_resumes.csv.
- Failed connections or malformed JSON are logged in parser_log.log.

### 2. Refresh Insights
- Reads existing data from parsed_resumes.csv (does not re-parse).
- Rebuilds insights and charts based on existing fields.
- Uses all available fields to generate combined text for analysis.

### 3. Training Models
- Uses the same combined resume text to train:
  - TF-IDF vectorizer
  - K-Means clustering model
- Produces clusters and summaries of similar resumes.
- Models are stored in memory during the Flask runtime.

## How Text Is Built for Insights

Each resume combines all available fields into one text block before analysis. This ensures that all skills, education, and experience are considered.

fields = ["summary", "skills", "education", "experience", "projects", "research", "relevant_classwork"]

texts = []
for _, row in df.iterrows():
    parts = []
    for c in fields:
        val = str(row.get(c, "")).strip()
        if val and val.lower() != "[]":
            parts.append(val)
    doc = " ".join(parts).strip()
    if len(doc.split()) > 2:
        texts.append(doc)

if not texts:
    return df, {
        "skills": {"labels": [], "values": []},
        "companies": {"labels": [], "values": []}
    }, {"summaries": []}

TF-IDF:
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(max_features=4000, stop_words="english")
X = vec.fit_transform(texts)

K-Means:
from sklearn.cluster import KMeans
import math

k = min(6, max(2, int(round(math.sqrt(len(texts))))))
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
labels = kmeans.fit_predict(X)

## Refresh Insights Behavior

The Refresh Insights button does not re-parse resumes. It reloads and re-analyzes the current parsed_resumes.csv file to rebuild:
- TF-IDF features
- Clusters
- Skill and company charts
- Summaries

Use Refresh Insights when you modify the CSV or adjust analysis code.

## Safety and Error Handling

When no text data exists or CSV is empty:
return df, {
    "skills": {"labels": [], "values": []},
    "companies": {"labels": [], "values": []}
}, {"summaries": []}

The app loads safely with an empty dashboard.

## Common Issues and Solutions

| Error | Cause | Fix |
|-------|--------|----|
| OSError: model requires more system memory | Model too large | Use smaller model (gemma:2b or llama3.2:1b) |
| charts is undefined | CSV empty or no usable text | Refresh CSV or ensure concatenation logic exists |
| ValueError: empty vocabulary | TF-IDF got no words | Ensure each resume has at least one non-empty text field |
| No module named 'PyPDF2' | Missing dependency | pip install PyPDF2 |
| ollama not recognized | PATH not set | Reopen terminal or add Ollama path to environment variables |
| Failed to connect to Ollama | Ollama not running | Run ollama serve in another terminal |
| CSV empty after parsing | Wrong folder path or concurrency issue | Ensure correct folder and set max_workers=1 in batch_parse |

## Default Fallbacks and Empty State Handling

All main functions (build_insights_from_csv, train_models_from_df, analyze_resume) include guards:
if not texts:
    return df, {
        "skills": {"labels": [], "values": []},
        "companies": {"labels": [], "values": []}
    }, {"summaries": []}

and Flask always passes charts and clusters to prevent UndefinedError.

## Memory and Performance

- Recommended max_workers=1 for Ollama to avoid connection timeouts.
- Limit parsing to small batches of resumes per run if testing.
- Each resume extraction can take 4-6 minutes depending on model size.

## Requirements

Example requirements.txt:
flask
pandas
scikit-learn
python-slugify
pdfplumber
docx2txt
tqdm
PyPDF2
pypdf
ollama

Install with:
pip install -r requirements.txt


## Future Improvements
- Store trained models persistently
- Add database support (SQLite or DuckDB)
- Add job title classification and keyword matching
- Support multiple local or remote LLM backends




