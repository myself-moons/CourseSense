"""
Text Preprocessing Pipeline — Coursera Course Reviews
========================================================
Input : coursera_reviews_sampled.csv  (raw reviews, 9000 rows)
Output: coursera_reviews_sampled_text_preprocessed.csv         (adds a clean_review column)

Steps applied, in order, and WHY each one is there:
1. Lowercase              -> "Great" and "great" should be treated as the same word
2. Remove URLs             -> a handful of reviews contain links; they add noise, no signal
3. Strip non-alphabetic    -> numbers/punctuation rarely help sentiment; removing them
                               shrinks the vocabulary and reduces sparsity for BoW/TF-IDF
4. Collapse whitespace     -> cleanup after the regex substitutions above
5. Tokenize                -> split the string into a list of words (nltk.word_tokenize
                               is smarter than str.split(), e.g. handles contractions)
6. Remove stopwords        -> drop high-frequency, low-information words ("the", "is", "a")
                               so the model focuses on words that actually carry meaning
7. Drop tokens <= 2 chars  -> leftover fragments after cleaning ("ll", "re") aren't useful
8. Lemmatize (verb + noun) -> reduce words to their dictionary root
                               e.g. "running"/"ran" -> "run", "courses" -> "course"
                               This shrinks the vocabulary further than stemming would,
                               while keeping the result a real word (unlike a stemmer).
9. Rejoin into a string    -> scikit-learn's CountVectorizer/TfidfVectorizer expect
                               plain strings, not token lists, as input
"""

import re
from pathlib import Path

import nltk
import pandas as pd

# Keep the script independent of the caller's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
NLTK_DATA_DIR = Path.home() / ".cache" / "coursesense" / "nltk_data"
NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
nltk.data.path.insert(0, str(NLTK_DATA_DIR))


def ensure_nltk_resources() -> None:
    """Download the small NLTK resource set once, if it is not available."""
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
    }
    for resource_path, package_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            if not nltk.download(package_name, download_dir=str(NLTK_DATA_DIR), quiet=True):
                raise RuntimeError(f"Could not download the NLTK resource: {package_name}")


ensure_nltk_resources()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Apply the full preprocessing pipeline to a single review and return
    a cleaned, space-joined string of tokens."""

    # 1. Lowercase everything
    text = str(text).lower()

    # 2. Remove URLs (http://..., www...)
    text = re.sub(r"http\S+|www\S+", " ", text)

    # 3. Keep only letters and spaces (drops digits, punctuation, emojis, etc.)
    text = re.sub(r"[^a-z\s]", " ", text)

    # 4. Collapse multiple spaces into one and trim ends
    text = re.sub(r"\s+", " ", text).strip()

    # 5. Tokenize into a list of words
    tokens = word_tokenize(text)

    # 6 & 7. Remove stopwords and very short leftover tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    # 8. Lemmatize — run twice with different POS tags since WordNetLemmatizer
    #    needs to know if a word is being treated as a verb or a noun
    tokens = [LEMMATIZER.lemmatize(t, pos="v") for t in tokens]   # verb forms: "learning"->"learn"
    tokens = [LEMMATIZER.lemmatize(t, pos="n") for t in tokens]   # noun forms: "courses"->"course"

    # 9. Rejoin into a single string
    return " ".join(tokens)


def main():
    input_path = DATA_DIR / "coursera_reviews_sampled.csv"
    output_path = DATA_DIR / "coursera_reviews_sampled_text_preprocessed.csv"
    df = pd.read_csv(input_path)

    # Apply the cleaning function to every review
    df["clean_review"] = df["review"].apply(clean_text)

    # Track how many words survived cleaning (useful for EDA / sanity-checking)
    df["clean_word_count"] = df["clean_review"].str.split().str.len()

    # Safety check: drop any review that became empty after cleaning
    before = len(df)
    df = df[df["clean_word_count"] > 0].reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows that became empty after cleaning")

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} cleaned rows -> {output_path}")

    # Show a few before/after examples
    print("\nSample before -> after:")
    for _, row in df.head(5).iterrows():
        print(f"  BEFORE: {row['review'][:80]}")
        print(f"  AFTER : {row['clean_review'][:80]}\n")


if __name__ == "__main__":
    main()
