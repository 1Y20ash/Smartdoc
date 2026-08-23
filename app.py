from pathlib import Path

import fitz
from docx import Document
from flask import Flask, flash, redirect, render_template, request, url_for

from src.classifier import load_classifier
from src.preprocessing import clean_text
from src.topic_model import get_document_topic_distribution, get_top_words_per_topic, load_topic_model


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
app.secret_key = "smartdoc-development-key"

vectorizer, lda_model = load_topic_model()
classifier = load_classifier()

# Human-readable interpretations of the topics discovered during LDA training.
TOPIC_NAMES = {
    1: "International Politics & Iraq",
    2: "International Affairs & Business Deals",
    3: "Government, Courts & Nuclear Affairs",
    4: "Sports",
    5: "Science, Space & Economy",
    6: "Conflict, Crime & Elections",
    7: "Business & Financial Markets",
    8: "Technology & Internet",
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text(file_path, extension):
    """Extract text from supported document formats."""
    if extension == "txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")

    if extension == "pdf":
        with fitz.open(file_path) as document:
            return "\n".join(page.get_text() for page in document)

    if extension == "docx":
        document = Document(file_path)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    raise ValueError("Unsupported document type.")


def analyze_document(text):
    """Run preprocessing, LDA topic modeling, and classification."""
    cleaned = clean_text(text)
    if not cleaned.strip():
        raise ValueError("The document does not contain readable text.")

    topic_distribution = get_document_topic_distribution(
        [cleaned], vectorizer, lda_model
    )[0]
    predicted_category = classifier.predict([topic_distribution])[0]

    topic_words = get_top_words_per_topic(vectorizer, lda_model, n_words=8)
    topic_results = []
    ranked_topics = topic_distribution.argsort()[::-1]

    for index in ranked_topics:
        topic_number = int(index + 1)
        topic_results.append(
            {
                "number": topic_number,
                "name": TOPIC_NAMES.get(topic_number, f"Topic {topic_number}"),
                "probability": float(topic_distribution[index]),
                "words": topic_words[topic_number],
            }
        )

    return {
        "category": predicted_category,
        "topics": topic_results,
        "cleaned_word_count": len(cleaned.split()),
    }


@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        uploaded_file = request.files.get("document")

        if not uploaded_file or not uploaded_file.filename:
            flash("Please choose a document to analyze.")
            return redirect(url_for("home"))

        if not allowed_file(uploaded_file.filename):
            flash("Supported formats: PDF, DOCX, and TXT.")
            return redirect(url_for("home"))

        extension = uploaded_file.filename.rsplit(".", 1)[1].lower()
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = Path(uploaded_file.filename).name
        file_path = UPLOAD_DIR / safe_name
        uploaded_file.save(file_path)

        try:
            text = extract_text(file_path, extension)
            result = analyze_document(text)
            result["filename"] = safe_name
        except Exception as exc:
            flash(f"Could not analyze the document: {exc}")
        finally:
            file_path.unlink(missing_ok=True)

    return render_template("index.html", result=result)


@app.errorhandler(413)
def request_too_large(_error):
    flash("File is too large. Maximum size is 10 MB.")
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
