"""
CourseSense — Personalized Skill Recommendation Engine (Colab-ready)
================================================================
Input : best_gru_model.keras (from gru_optuna_tuning.py), engineered_features.csv,
        gru_sequences.npz, skill_vocab.json
Output: printed recommendations; recommend_next_skills() is reusable for a dashboard/app

THE CORE IDEA
--------------------------------------------
The trained GRU predicts P(correct | student's history so far, this specific
interaction). That's exactly the building block a recommender needs — it's a
per-skill MASTERY ESTIMATE, conditioned on one particular student's own
history. To recommend, we ask the model a "what if" question for every skill:

    "If this student attempted skill X next, what's their predicted
     probability of getting it right?"

We do this by taking the student's real interaction history, appending ONE
hypothetical next step for each of the 93 skills (with correctly-updated
opportunity count, prior success/failure count for that skill, etc.), running
all 93 hypothetical continuations through the model in a single batch, and
reading off the predicted probability at that final position for each one.

TWO RECOMMENDATION STRATEGIES
--------------------------------------------
  'remediate' — recommend the skills with the LOWEST predicted probability.
                Use case: "what does this student most need to work on?"
  'zpd'       — recommend skills near ~60% predicted probability: challenging
                but achievable (Zone of Proximal Development / "desirable
                difficulty" from educational psychology). Use case: "what
                should this student practice next to keep learning without
                frustration or boredom?"
Both are legitimate; which one to surface depends on the product goal —
remediation is more diagnostic/instructor-facing, ZPD is more like a
practice-queue for the student themselves.
"""

import json
import pickle
import numpy as np
import pandas as pd
from tensorflow import keras

# ------------------------------------------------------------------
# STEP 1 — Load the trained model and supporting data
# ------------------------------------------------------------------
model = keras.models.load_model("best_gru_model.keras")

df = pd.read_csv("engineered_features.csv")
data = np.load("gru_sequences.npz")
skill_arr, correct_arr, interaction_arr, seq_lengths = (
    data["skill_arr"], data["correct_arr"], data["interaction_arr"], data["seq_lengths"]
)
num_skills = int(data["num_skills"])
max_len = skill_arr.shape[1]
sorted_user_ids = np.sort(df["user_id"].unique())

START_TOKEN = 2 * num_skills + 1
PAD_TOKEN = 2 * num_skills

with open("skill_vocab.json") as f:
    skill_to_idx = json.load(f)
idx_to_skill = {v: k for k, v in skill_to_idx.items()}

# For a DEPLOYED recommender (not a held-out evaluation), skill difficulty is
# computed from ALL available data — there's no leakage concern here since
# we're not scoring model performance, just serving real recommendations.
skill_difficulty_map = df.groupby("clean_skill_name")["correct"].mean().to_dict()
skill_difficulty_by_idx = np.array([skill_difficulty_map[idx_to_skill[i]] for i in range(num_skills)])

SIDE_FEAT_COLS = ["opportunity", "prior_success_count", "prior_failure_count",
                   "student_rolling_accuracy", "log_ms_first_response", "log_overlap_time"]
# NOTE: this column order must match gru_optuna_tuning.py exactly (base features,
# then skill_difficulty appended last) — that's the order the model was trained on.


def get_student_row_idx(user_id):
    """Convert a real user_id into its row index in the sequence arrays."""
    matches = np.where(sorted_user_ids == user_id)[0]
    if len(matches) == 0:
        raise ValueError(f"user_id {user_id} not found in this dataset")
    return int(matches[0])


def recommend_next_skills(student, top_k=5, strategy="remediate"):
    """
    student: either a row index (int, 0..343) or a real user_id present in the data
    top_k:   how many skills to recommend
    strategy: 'remediate' (weakest skills) or 'zpd' (challenging but achievable, ~60%)

    Returns a DataFrame: skill, predicted_success_prob, times_practiced_before
    """
    student_row_idx = student if isinstance(student, (int, np.integer)) and student < len(sorted_user_ids) and student not in sorted_user_ids else get_student_row_idx(student)
    # (the check above prefers treating small ints as row indices; pass the user_id
    #  explicitly as e.g. int(user_id) if your user_ids happen to overlap row-index range)

    L = int(seq_lengths[student_row_idx])
    hist_skills = skill_arr[student_row_idx, :L]
    hist_correct = correct_arr[student_row_idx, :L]
    hist_interactions = interaction_arr[student_row_idx, :L]

    # Tally the student's current per-skill history
    opp_count = {s: 0 for s in range(num_skills)}
    succ_count = {s: 0 for s in range(num_skills)}
    fail_count = {s: 0 for s in range(num_skills)}
    for s, c in zip(hist_skills, hist_correct):
        opp_count[s] += 1
        if c == 1:
            succ_count[s] += 1
        else:
            fail_count[s] += 1
    rolling_acc = hist_correct.sum() / L

    student_uid = sorted_user_ids[student_row_idx]
    student_rows = df[df["user_id"] == student_uid]
    avg_log_rt = student_rows["log_ms_first_response"].mean()
    avg_log_ot = student_rows["log_overlap_time"].mean()
    hist_side = student_rows[SIDE_FEAT_COLS].values
    hist_diff = np.array([skill_difficulty_by_idx[s] for s in hist_skills])

    # Build one hypothetical continuation per candidate skill, batched together
    X_batch = np.full((num_skills, max_len), PAD_TOKEN, dtype=np.int32)
    side_batch = np.zeros((num_skills, max_len, len(SIDE_FEAT_COLS) + 1), dtype=np.float32)

    for cand in range(num_skills):
        X_batch[cand, 0] = START_TOKEN
        X_batch[cand, 1:L] = hist_interactions[:L - 1]
        X_batch[cand, L] = hist_interactions[L - 1]   # last real interaction feeds the hypothetical step

        side_batch[cand, :L, :len(SIDE_FEAT_COLS)] = hist_side
        side_batch[cand, :L, len(SIDE_FEAT_COLS)] = hist_diff

        side_batch[cand, L, 0] = opp_count[cand] + 1          # opportunity
        side_batch[cand, L, 1] = succ_count[cand]              # prior_success_count
        side_batch[cand, L, 2] = fail_count[cand]               # prior_failure_count
        side_batch[cand, L, 3] = rolling_acc                    # student_rolling_accuracy (current)
        side_batch[cand, L, 4] = avg_log_rt                     # proxy: student's own average response time
        side_batch[cand, L, 5] = avg_log_ot                     # proxy: student's own average overlap time
        side_batch[cand, L, 6] = skill_difficulty_by_idx[cand]  # skill_difficulty of the candidate

    preds = model.predict([X_batch, side_batch], verbose=0)[:, L, 0]

    results = pd.DataFrame({
        "skill": [idx_to_skill[i] for i in range(num_skills)],
        "predicted_success_prob": preds,
        "times_practiced_before": [opp_count[i] for i in range(num_skills)],
    }).sort_values("predicted_success_prob").reset_index(drop=True)

    if strategy == "remediate":
        return results.head(top_k)
    elif strategy == "zpd":
        results = results.copy()
        results["dist_from_target"] = (results["predicted_success_prob"] - 0.60).abs()
        return results.sort_values("dist_from_target").head(top_k).drop(columns="dist_from_target")
    else:
        raise ValueError("strategy must be 'remediate' or 'zpd'")


# ------------------------------------------------------------------
# STEP 2 — Demo: recommend for a couple of example students
# ------------------------------------------------------------------
if __name__ == "__main__":
    for demo_idx in [0, 5, 20]:
        uid = sorted_user_ids[demo_idx]
        print(f"\n{'='*60}\nStudent user_id={uid} (row {demo_idx}, "
              f"{seq_lengths[demo_idx]} interactions so far)\n{'='*60}")

        print("\n-- Remediation (skills needing the most work) --")
        print(recommend_next_skills(demo_idx, top_k=5, strategy="remediate").to_string(index=False))

        print("\n-- ZPD (challenging but achievable, ~60% predicted success) --")
        print(recommend_next_skills(demo_idx, top_k=5, strategy="zpd").to_string(index=False))