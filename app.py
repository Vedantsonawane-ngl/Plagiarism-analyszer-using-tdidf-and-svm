"""
Enhanced Plagiarism Checker — Flask Backend
Adds: plagiarism %, sentence highlighting, file upload, scan history, dashboard stats
"""

from flask import Flask, render_template, request, jsonify, session
import pickle
import os
import json
import re
import csv
from datetime import datetime

# ── Optional file-parsing libraries ─────────────────────────────────────────
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(24)          # needed for session
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB upload cap

# ── Load ML artefacts ────────────────────────────────────────────────────────
model           = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# ── Load reference corpus from dataset.csv ──────────────────────────────────
REFERENCE_SENTENCES = []
try:
    with open('dataset.csv', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('source_text'):
                REFERENCE_SENTENCES.append(row['source_text'].strip())
            if row.get('plagiarized_text'):
                REFERENCE_SENTENCES.append(row['plagiarized_text'].strip())
except FileNotFoundError:
    pass

# ── Persistent scan history (JSON file) ─────────────────────────────────────
HISTORY_FILE = 'scan_history.json'

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []

def save_history(entry):
    history = load_history()
    history.insert(0, entry)
    history = history[:100]            # keep last 100 scans
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

# ── Core detection helpers ───────────────────────────────────────────────────
def split_sentences(text):
    """Naive sentence splitter."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def compute_plagiarism_percentage(input_text):
    """
    Returns a 0-100 float representing overall similarity to the reference
    corpus using cosine similarity on TF-IDF vectors.
    """
    if not REFERENCE_SENTENCES:
        # Fall back to model's raw decision probability
        vec = tfidf_vectorizer.transform([input_text])
        proba = model.predict_proba(vec)[0]
        return round(float(proba[1]) * 100, 1)

    input_vec  = tfidf_vectorizer.transform([input_text])
    ref_matrix = tfidf_vectorizer.transform(REFERENCE_SENTENCES)
    sims = cosine_similarity(input_vec, ref_matrix)[0]
    return round(float(np.max(sims)) * 100, 1)


def highlight_sentences(input_text, threshold=0.55):
    """
    Returns a list of dicts: {sentence, plagiarized (bool), score, matched_source}.
    Each sentence is compared to the reference corpus individually.
    """
    sentences = split_sentences(input_text)
    results   = []

    for sent in sentences:
        if not REFERENCE_SENTENCES:
            # Use model directly
            vec    = tfidf_vectorizer.transform([sent])
            label  = int(model.predict(vec)[0])
            score  = 0.0
            source = ""
        else:
            vec    = tfidf_vectorizer.transform([sent])
            ref    = tfidf_vectorizer.transform(REFERENCE_SENTENCES)
            sims   = cosine_similarity(vec, ref)[0]
            best_i = int(np.argmax(sims))
            score  = float(sims[best_i])
            label  = 1 if score >= threshold else 0
            source = REFERENCE_SENTENCES[best_i] if label else ""

        results.append({
            "sentence"      : sent,
            "plagiarized"   : bool(label),
            "score"         : round(score * 100, 1),
            "matched_source": source
        })
    return results


def extract_text_from_pdf(file_obj):
    if not PDF_SUPPORT:
        return None, "pdfplumber not installed. Run: pip install pdfplumber"
    text = ""
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text.strip(), None


def extract_text_from_docx(file_obj):
    if not DOCX_SUPPORT:
        return None, "python-docx not installed. Run: pip install python-docx"
    doc   = DocxDocument(file_obj)
    text  = "\n".join(p.text for p in doc.paragraphs)
    return text.strip(), None

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    history = load_history()
    total   = len(history)
    avg_pct = round(sum(h['percentage'] for h in history) / total, 1) if total else 0
    return render_template('dashboard.html', history=history, total=total, avg_pct=avg_pct)


@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    """
    Accepts JSON or form-data (text or file).
    Returns JSON with: percentage, label, highlighted_sentences, timestamp.
    """
    input_text  = ""
    error_msg   = None

    # ── 1. Text from JSON body (API call from JS fetch) ──────────────────
    if request.is_json:
        data       = request.get_json()
        input_text = data.get('text', '').strip()

    # ── 2. File upload ───────────────────────────────────────────────────
    elif 'file' in request.files and request.files['file'].filename:
        f    = request.files['file']
        name = f.filename.lower()

        if name.endswith('.pdf'):
            input_text, error_msg = extract_text_from_pdf(f)
        elif name.endswith('.docx'):
            input_text, error_msg = extract_text_from_docx(f)
        elif name.endswith('.txt'):
            input_text = f.read().decode('utf-8', errors='ignore').strip()
        else:
            error_msg = "Unsupported file type. Please upload PDF, DOCX, or TXT."

    # ── 3. Plain form text ───────────────────────────────────────────────
    else:
        input_text = request.form.get('text', '').strip()

    # ── Guard clauses ────────────────────────────────────────────────────
    if error_msg:
        return jsonify({"error": error_msg}), 400

    if not input_text:
        return jsonify({"error": "No text provided."}), 400

    if len(input_text) < 20:
        return jsonify({"error": "Text too short for analysis (min 20 characters)."}), 400

    # ── Run analysis ─────────────────────────────────────────────────────
    percentage          = compute_plagiarism_percentage(input_text)
    highlighted         = highlight_sentences(input_text)
    plagiarised_count   = sum(1 for s in highlighted if s['plagiarized'])
    label               = "Plagiarism Detected" if percentage >= 30 else "No Plagiarism Detected"
    timestamp           = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Persist to history ───────────────────────────────────────────────
    save_history({
        "timestamp"      : timestamp,
        "snippet"        : input_text[:120] + ("…" if len(input_text) > 120 else ""),
        "percentage"     : percentage,
        "label"          : label,
        "sentence_count" : len(highlighted),
        "flagged_count"  : plagiarised_count,
    })

    return jsonify({
        "percentage"           : percentage,
        "label"                : label,
        "highlighted_sentences": highlighted,
        "timestamp"            : timestamp,
        "sentence_count"       : len(highlighted),
        "flagged_count"        : plagiarised_count,
        "word_count"           : len(input_text.split()),
    })


@app.route('/api/history')
def api_history():
    return jsonify(load_history())


@app.route('/api/stats')
def api_stats():
    history = load_history()
    total   = len(history)
    avg_pct = round(sum(h['percentage'] for h in history) / total, 1) if total else 0
    high    = sum(1 for h in history if h['percentage'] >= 70)
    med     = sum(1 for h in history if 30 <= h['percentage'] < 70)
    low     = total - high - med
    return jsonify({
        "total": total, "avg_pct": avg_pct,
        "high": high, "med": med, "low": low,
        "recent": history[:7]
    })


if __name__ == "__main__":
    app.run(debug=True)
