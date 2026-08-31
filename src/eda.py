# %% [markdown]
# # CourseSense — Exploratory Data Analysis
# Dataset: `assistments_sample_AFTER.csv` (15,061 interactions, 344 students, 94 skills)
#
# Run each `# %%` block as its own cell in Colab / Jupyter.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_csv("/workspaces/CourseSense/data/assistments_sample_AFTER.csv")
print("Shape:", df.shape)
print(df.dtypes)

# %% [markdown]
# ## 1. Basic integrity checks

# %%
print("Missing values per column:\n", df.isna().sum())
print("\nUnique students:", df["user_id"].nunique())
print("Unique skills:", df["clean_skill_name"].nunique())

# %% [markdown]
# ## 2. Target distribution — correct vs. incorrect

# %%
target_counts = df["correct"].value_counts().sort_index()
print(target_counts)
print("Correct rate:", df["correct"].mean().round(3))

plt.figure(figsize=(5, 4))
plt.bar(["Incorrect (0)", "Correct (1)"], target_counts.values, color=["#e07a5f", "#3d9970"])
plt.title("Target class distribution: Correct vs Incorrect")
plt.ylabel("Number of interactions")
plt.tight_layout()
plt.savefig("eda_target_distribution.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Sequence length per student
# A GRU needs to see real variation in sequence length — this checks that we have it.

# %%
seq_len = df.groupby("user_id").size()
print(seq_len.describe())

bins = [15, 25, 35, 45, 60, 80, 121]
labels = ["15-24", "25-34", "35-44", "45-59", "60-79", "80-120"]
bucket = pd.cut(seq_len, bins=bins, labels=labels, right=False, include_lowest=True)
bucket_counts = bucket.value_counts().reindex(labels)
print(bucket_counts)

plt.figure(figsize=(6, 4))
plt.bar(bucket_counts.index, bucket_counts.values, color="#457b9d")
plt.title("Student sequence length distribution")
plt.xlabel("Interactions per student")
plt.ylabel("Number of students")
plt.tight_layout()
plt.savefig("eda_sequence_length.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Skill frequency — which skills are practiced most?

# %%
top_skills = df["clean_skill_name"].value_counts().head(10)
print(top_skills)

plt.figure(figsize=(8, 5))
plt.barh(top_skills.index[::-1], top_skills.values[::-1], color="#264653")
plt.title("Top 10 most-practiced skills")
plt.xlabel("Number of attempts")
plt.tight_layout()
plt.savefig("eda_top_skills.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Correctness rate by skill (difficulty spread)
# Restricted to skills with >= 100 attempts so the rate is stable.

# %%
skill_counts = df["clean_skill_name"].value_counts()
frequent_skills = skill_counts[skill_counts >= 100].index
skill_acc = (
    df[df["clean_skill_name"].isin(frequent_skills)]
    .groupby("clean_skill_name")["correct"]
    .mean()
    .sort_values()
)

print("Hardest 5 skills:\n", skill_acc.head(5))
print("\nEasiest 5 skills:\n", skill_acc.tail(5))

hardest_easiest = pd.concat([skill_acc.head(5), skill_acc.tail(5)])
plt.figure(figsize=(8, 5))
plt.barh(hardest_easiest.index, hardest_easiest.values, color="#e76f51")
plt.title("Hardest vs. easiest skills (min. 100 attempts)")
plt.xlabel("Correctness rate")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig("eda_skill_difficulty.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Learning curve — does accuracy improve with practice?
# For each student-skill pair, number attempts in chronological order (1st time seeing
# the skill, 2nd time, ...), then average correctness at each opportunity number.

# %%
df_sorted = df.sort_values(["user_id", "clean_skill_name", "order_id"]).copy()
df_sorted["opportunity"] = df_sorted.groupby(["user_id", "clean_skill_name"]).cumcount() + 1

lc = df_sorted[df_sorted["opportunity"] <= 10].groupby("opportunity")["correct"].agg(["mean", "count"])
print(lc)

plt.figure(figsize=(7, 4))
plt.plot(lc.index, lc["mean"], marker="o", color="#2a9d8f")
plt.title("Learning curve: accuracy by practice attempt on a skill")
plt.xlabel("Opportunity number (nth time practicing this skill)")
plt.ylabel("Correctness rate")
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("eda_learning_curve.png", dpi=150)
plt.show()

# %% [markdown]
# ## 7. Attempt count & hint usage
# Checking these because they're candidate engineered features — and because hint
# usage turns out to leak the answer (see findings below).

# %%
print("Attempt count distribution:\n", df["attempt_count"].value_counts().sort_index().head(10))
print("\nMean attempt_count:", df["attempt_count"].mean().round(3))

pct_hint = (df["hint_count"] > 0).mean()
print(f"\n% interactions with >=1 hint used: {pct_hint:.3f}")
print("Correctness rate, hint used vs not:\n", df.groupby(df["hint_count"] > 0)["correct"].mean())

# %% [markdown]
# ## 8. Skill-name vocabulary (from `clean_skill_name`)
# This is the text feature engineering will build on (BoW / TF-IDF / Word2Vec).

# %%
all_words_weighted = " ".join(df["clean_skill_name"]).split()
top_words = Counter(all_words_weighted).most_common(15)
print("Most common words across skill occurrences (frequency-weighted):")
for word, count in top_words:
    print(f"  {word:<15} {count}")

# %% [markdown]
# ## Findings & Next Steps
#
# **Dataset integrity:** 15,061 interactions, 344 students, 94 skills, zero missing
# values across all 13 columns.
#
# **Target balance:** 61.4% correct / 38.6% incorrect — moderately imbalanced but
# workable without resampling.
#
# **Sequence variation:** student sequences range from 15 to 120 interactions
# (median 34), giving the GRU real variety to learn from rather than a flat,
# uniform input length.
#
# **Skill frequency is long-tailed:** the top skill alone makes up ~10.6% of all
# rows; many of the 94 skills have far fewer examples, which limits how reliably
# we can model some individual skills in isolation.
#
# **Skill difficulty varies widely:** correctness rate ranges from 36% ("finding
# percent") to 87% ("mode") among skills with at least 100 attempts. This is a
# strong candidate feature — and something the GRU should be able to pick up
# implicitly through its skill embeddings.
#
# **A real learning-curve effect exists:** accuracy jumps from 52.4% on a
# student's first attempt at a skill to 63–66% on later attempts, then plateaus.
# This is direct evidence that a student's history on a skill is predictive of
# their future performance — the core premise behind "personalized learning
# recommendation."
#
# **Data leakage risk — `hint_count`:** interactions where a hint was used have a
# correctness rate of ~0.04%, vs. 74% with no hint. This isn't a "hints hurt
# learning" signal so much as hint requests and wrong answers being recorded
# together on the same attempt. `hint_count` should be excluded or carefully
# re-engineered (e.g. hint usage on *previous* attempts only) rather than fed in
# directly, or the model will just learn to detect hints.
#
# **`attempt_count` looks safer:** 77% of interactions were solved in a single
# attempt, with a long tail out to 9+ attempts — a reasonable engineered feature
# for the classical baselines.
#
# **Skill-name vocabulary:** dominated by math-topic words (fraction, addition,
# subtraction, percent, decimal, equation, probability) — a sensible, non-trivial
# vocabulary size for TF-IDF, though likely too small on its own for a strong
# Word2Vec model (worth using pretrained GloVe vectors instead, or treating
# Word2Vec as a smaller demonstration piece alongside TF-IDF).
#
# ### Next steps
# 1. **Feature engineering** — build the per-student skill-ID sequence for the
#    GRU (Embedding layer), and BoW/TF-IDF (+ optionally Word2Vec/GloVe) on
#    `clean_skill_name` for the classical baselines.
# 2. **Engineer safe features** — rolling per-student accuracy so far, opportunity
#    count per skill, skill difficulty (global correctness rate), attempt_count.
#    Exclude raw `hint_count` or lag it by one step.
# 3. **Train classical baselines** (Naive Bayes, Logistic Regression, Random
#    Forest, SVM, or XGBoost) on the engineered features to predict `correct`.
# 4. **Train the GRU** on the true chronological sequence per student to predict
#    the next interaction's correctness.
# 5. **Compare** classical baselines vs. GRU on Accuracy, Precision, Recall,
#    F1, Confusion Matrix, and ROC-AUC.