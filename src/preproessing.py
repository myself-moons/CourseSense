"""
CourseSense — Dataset Build & Text Preprocessing (Colab-ready)
================================================================
Sequential Student Interaction Modeling with GRU Networks
Dataset: ASSISTments 2009-2010 Skill-Builder data

Run this top to bottom in Google Colab. It will:
  1. Download the full ASSISTments 2009-2010 dataset (~525k rows) from GitHub
  2. Filter + sample it down to a manageable, GRU-friendly sequential dataset
     (~15,000 interaction rows, keeping each student's full sequence intact)
  3. Save the BEFORE file (raw sampled data)
  4. Apply text preprocessing to the `skill_name` field (this is our NLP step —
     skill names are short text like "Addition and Subtraction Fractions" that
     BoW/TF-IDF/Word2Vec can meaningfully be built on top of, later)
  5. Save the AFTER file (with a new `clean_skill_name` column)

Why this dataset fits the project title:
  Each row = one student's attempt at one skill, in chronological order
  (`order_id`). A GRU can read a student's sequence of attempts and predict
  whether they'll get the NEXT one right — this is the actual mechanism
  behind "personalized learning recommendation": recommend what a student
  should practice next, based on a model of their current mastery.
"""

import re
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# STEP 0 — Setup
# ------------------------------------------------------------------
# !pip install nltk -q   # uncomment if nltk isn't already available in your Colab runtime

import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("punkt_tab")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

SEED = 42
np.random.seed(SEED)

# ------------------------------------------------------------------
# STEP 1 — Download the full dataset
# ------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/GaoSida/DKT/master/Assistments/skill_builder_data.csv"

print("Downloading dataset...")
df = pd.read_csv(DATA_URL, encoding="ISO-8859-1", low_memory=False)
print(f"Full dataset shape: {df.shape}")

# Keep only the columns relevant to sequential modeling + the text field for NLP
KEEP_COLS = [
    "order_id", "assignment_id", "user_id", "problem_id", "skill_id", "skill_name",
    "correct", "attempt_count", "hint_count", "hint_total", "ms_first_response", "overlap_time",
]
df = df[KEEP_COLS]

# Drop rows with no skill_name (no text to work with) or no skill_id
df = df.dropna(subset=["skill_name", "skill_id"]).reset_index(drop=True)
print(f"After dropping rows with missing skill_name/skill_id: {df.shape}")
print(f"Unique students: {df['user_id'].nunique()} | Unique skills: {df['skill_name'].nunique()}")

# ------------------------------------------------------------------
# STEP 2 — Sample down to ~15,000 rows, keeping full student sequences intact
# ------------------------------------------------------------------
# We must never split a student's sequence — a GRU needs each student's
# interactions to stay complete and in order. So we sample whole students,
# not random rows.

counts = df.groupby("user_id").size()

# Keep students with a reasonable sequence length: long enough to be a real
# sequence, short enough that no single student dominates the sample
MIN_LEN, MAX_LEN = 15, 120
eligible_students = counts[(counts >= MIN_LEN) & (counts <= MAX_LEN)].index.to_numpy().copy()

rng = np.random.default_rng(SEED)
rng.shuffle(eligible_students)

TARGET_ROWS = 15000
chosen, total = [], 0
for sid in eligible_students:
    n = counts[sid]
    if total + n > TARGET_ROWS + 1000:   # don't overshoot by too much
        continue
    chosen.append(sid)
    total += n
    if total >= TARGET_ROWS:
        break

print(f"\nChosen students: {len(chosen)} | Total interaction rows: {total}")

sample = df[df["user_id"].isin(chosen)].copy()
sample = sample.sort_values(["user_id", "order_id"]).reset_index(drop=True)  # true chronological order

print(f"Final sample shape: {sample.shape}")
print(f"Students: {sample['user_id'].nunique()} | Unique skills: {sample['skill_name'].nunique()}")
print(f"Overall correct rate: {sample['correct'].mean():.3f}")
print("\nInteractions-per-student summary:")
print(sample.groupby("user_id").size().describe())

# ------------------------------------------------------------------
# STEP 3 — Save the BEFORE file (raw sampled data)
# ------------------------------------------------------------------
sample.to_csv("assistments_sample_BEFORE.csv", index=False)
print("\nSaved assistments_sample_BEFORE.csv")

# ------------------------------------------------------------------
# STEP 4 — Text preprocessing on skill_name
# ------------------------------------------------------------------
# Skill names are short phrases like "Addition and Subtraction Fractions".
# We clean them the same way we'd clean any text field, so that later
# feature engineering (BoW / TF-IDF / Word2Vec) works on a clean vocabulary
# instead of raw, inconsistent strings.
#
#   1. Lowercase              -> case-insensitive matching
#   2. Strip non-alphabetic   -> remove stray punctuation/numbers
#   3. Tokenize               -> split into individual words
#   4. Remove stopwords       -> drop "and", "of", etc. (no signal, wastes vocabulary)
#   5. Lemmatize (noun form)  -> "Fractions" -> "fraction", so plural/singular
#                                 forms of the same skill collapse together

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_skill_name(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t, pos="n") for t in tokens]
    return " ".join(tokens) if tokens else text  # fallback: keep original if everything got stripped


after = sample.copy()
after["clean_skill_name"] = after["skill_name"].apply(clean_skill_name)

print(f"\nUnique raw skill_name: {sample['skill_name'].nunique()}")
print(f"Unique clean_skill_name: {after['clean_skill_name'].nunique()}")
print("\nSample before -> after:")
print(after[["skill_name", "clean_skill_name"]].drop_duplicates().head(15).to_string())

# ------------------------------------------------------------------
# STEP 5 — Save the AFTER file
# ------------------------------------------------------------------
after.to_csv("assistments_sample_AFTER.csv", index=False)
print("\nSaved assistments_sample_AFTER.csv")
print(f"Final shape: {after.shape}")