# AI-Powered FAQ Chatbot with Semantic Search (SGMCOE)

A lightweight Flask chatbot that answers college FAQs by comparing the
*meaning/word-usage* of a question (via TF-IDF + Cosine Similarity),
not just exact word matching — with an interactive chat UI.

## Folder structure
```
faq_chatbot/
├── app.py              # Flask backend + NLP (TF-IDF) + similarity search
├── faqs.json           # FAQ data (53 Q&A pairs, tagged by category)
├── requirements.txt    # Python dependencies (lightweight, no PyTorch)
├── templates/
│   └── index.html      # Chat page (category tabs + suggestions)
└── static/
    ├── style.css        # Interactive chat UI styling
    └── script.js        # Chat logic, typing indicator, tab filtering
```

## How it works (in plain words)
1. **faqs.json** stores all questions, answers, and a category tag.
2. **Preprocessing**: each question is lowercased, cleaned, and lemmatized
   using NLTK (e.g. "running" → "run").
3. **Vocabulary + TF-IDF**: we collect every unique word across all FAQ
   questions and turn each question into a TF-IDF vector — a list of
   numbers where common words (like "the", "college") get a low weight
   and distinctive words (like "hostel", "placement") get a high weight.
4. **User query**: the typed question goes through the same cleaning +
   TF-IDF steps, using the same vocabulary.
5. **Matching**: a small NumPy `cosine_similarity` function compares the
   user's vector against every FAQ vector. The FAQ with the highest
   score wins.
6. **Answering**: if the best score is above the threshold (0.25), the
   matching answer + a confidence % is returned. Otherwise, a fallback
   "couldn't find an answer" message is shown.

This uses only `flask`, `nltk`, and `numpy` — no PyTorch or scikit-learn,
so it installs cleanly even on brand-new Python versions.

## Interactive UI features
- Category tabs (Admission, Courses, Fees, Facilities, Placements, etc.)
  to filter suggested questions
- Clickable suggestion chips so users know what to ask
- Typing indicator animation while the bot "thinks"
- Animated confidence bar under each bot answer
- Clear chat button
- Smooth message animations, mobile-responsive layout

## Setup

```bash
cd faq_chatbot
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The first run downloads a small amount of NLTK data (a few MB) — needs
internet once. After that, it works fully offline.

Open **http://127.0.0.1:5000** in your browser.

## Notes for your project report / viva
- Threshold value is set in `app.py` as `SIMILARITY_THRESHOLD = 0.25`.
  Increase it for stricter matching, decrease it to accept looser matches.
- Categories are read directly from the `category` field in `faqs.json`
  and used to build both the tabs and the suggestion chips.
- To add more FAQs, just add more `{"category": ..., "question": ...,
  "answer": ...}` entries to `faqs.json` — the vocabulary and TF-IDF
  matrix rebuild automatically each time the app starts.
