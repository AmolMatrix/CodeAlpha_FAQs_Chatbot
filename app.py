"""
AI-Powered FAQ Chatbot with Semantic Search (lightweight edition)
-------------------------------------------------------------------
A simple Flask app that answers user questions by comparing the
meaning of the question with a stored list of FAQs, using classic
NLP (TF-IDF + Cosine Similarity) instead of a heavy deep-learning
model. This avoids large dependencies like sentence-transformers /
PyTorch that can fail to install on very new Python versions.

How it works (step by step):
1. On startup, we load faqs.json (list of question/answer pairs).
2. We clean each FAQ question (lowercase, remove symbols, lemmatize)
   using NLTK.
3. We build a vocabulary from all FAQ questions and convert every
   question into a TF-IDF vector (a list of numbers where common
   words get low weight and distinctive words get high weight).
4. When a user sends a message, it goes through the same cleaning +
   TF-IDF steps, using the SAME vocabulary built in step 3.
5. We compare the user's vector with every FAQ vector using Cosine
   Similarity (a score between 0 and 1 showing how close two pieces
   of text are in meaning/word-usage).
6. We pick the FAQ with the highest similarity score. If that score
   is above a threshold, we return its answer + confidence score.
   Otherwise we return a fallback "I don't know" message.
"""

import json
import os
import re

import numpy as np
from flask import Flask, jsonify, render_template, request

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------
# 1. One-time NLTK setup (downloads small language data files)
# ---------------------------------------------------------------------
NLTK_PACKAGES = ["punkt", "punkt_tab", "wordnet", "omw-1.4"]
for pkg in NLTK_PACKAGES:
    try:
        nltk.data.find(f"tokenizers/{pkg}") if "punkt" in pkg else nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------------------------
# 2. Config
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_FILE = os.path.join(BASE_DIR, "faqs.json")
SIMILARITY_THRESHOLD = 0.25  # below this, we treat the question as "unknown"

app = Flask(__name__)


# ---------------------------------------------------------------------
# 3. Text preprocessing (cleaning, tokenization, lemmatization)
# ---------------------------------------------------------------------
def preprocess_text(text: str):
    """Lowercase, remove symbols, tokenize and lemmatize the text.
    Returns a list of clean word tokens."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(tok) for tok in tokens]


# ---------------------------------------------------------------------
# 4. Load FAQs and build a simple TF-IDF matrix (no external ML library)
# ---------------------------------------------------------------------
with open(FAQ_FILE, "r", encoding="utf-8") as f:
    faqs = json.load(f)

faq_tokens = [preprocess_text(item["question"]) for item in faqs]

# Build vocabulary: every unique word across all FAQ questions
vocab = sorted({tok for tokens in faq_tokens for tok in tokens})
word_to_index = {word: i for i, word in enumerate(vocab)}


def term_frequency(tokens):
    """TF vector: how often each vocabulary word appears in `tokens`."""
    vec = np.zeros(len(vocab))
    for tok in tokens:
        if tok in word_to_index:
            vec[word_to_index[tok]] += 1
    if len(tokens) > 0:
        vec = vec / len(tokens)
    return vec


def inverse_document_frequency(all_tokens):
    """IDF: words that appear in fewer FAQs get a higher weight,
    words that appear in almost every FAQ (like 'college') get a
    lower weight."""
    n_docs = len(all_tokens)
    doc_freq = np.zeros(len(vocab))
    for tokens in all_tokens:
        for tok in set(tokens):
            if tok in word_to_index:
                doc_freq[word_to_index[tok]] += 1
    return np.log((n_docs + 1) / (doc_freq + 1)) + 1


idf = inverse_document_frequency(faq_tokens)
faq_vectors = np.array([term_frequency(tokens) for tokens in faq_tokens]) * idf

print(f"Ready! {len(faqs)} FAQs loaded, vocabulary size: {len(vocab)}.")


# ---------------------------------------------------------------------
# 5. Core matching logic
# ---------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between one vector `a` (shape: [dim]) and many
    vectors `b` (shape: [n, dim]). Returns an array of n scores."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b, axis=1)
    denom = (a_norm * b_norm)
    denom[denom == 0] = 1e-10  # avoid divide-by-zero for empty vectors
    return (b @ a) / denom


def find_best_answer(user_question: str):
    tokens = preprocess_text(user_question)
    user_vector = term_frequency(tokens) * idf

    scores = cosine_similarity(user_vector, faq_vectors)
    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])

    if best_score >= SIMILARITY_THRESHOLD:
        return {
            "answer": faqs[best_index]["answer"],
            "matched_question": faqs[best_index]["question"],
            "category": faqs[best_index].get("category", "General"),
            "confidence": round(best_score * 100, 1),
            "found": True,
        }
    else:
        return {
            "answer": "Sorry, I couldn't find a suitable answer. Please contact support or rephrase your question.",
            "matched_question": None,
            "category": None,
            "confidence": round(best_score * 100, 1),
            "found": False,
        }


# ---------------------------------------------------------------------
# 6. Routes
# ---------------------------------------------------------------------
@app.route("/")
def index():
    categories = {}
    for item in faqs:
        cat = item.get("category", "General")
        if cat == "General":
            continue
        categories.setdefault(cat, []).append(item["question"])
    return render_template("index.html", categories=categories)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Please type a question."}), 400

    result = find_best_answer(user_message)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
