"""
CourseSense — Optuna-Tuned GRU with Cross-Validation (Colab-ready)
================================================================
Input : engineered_features.csv, gru_sequences.npz
Output: best_gru_model.keras, optuna_study.pkl, final_gru_results.json

WHAT CHANGED FROM THE BASELINE GRU, AND WHY
--------------------------------------------
The baseline GRU only saw the raw sequence of (skill, correct) pairs and
scored ROC-AUC 0.76 — well below the classical models' 0.93+, because the
classical models got rich engineered features (skill difficulty, prior
success/failure counts, response time) that the baseline GRU didn't have
access to at all.

This version fixes that gap by feeding the SAME engineered features into
the GRU at every timestep, alongside its own recurrent memory of the
student's history. Concretely, to predict whether a student gets THIS
interaction right, the model combines:
  (a) a GRU hidden state built from the student's interactions strictly
      BEFORE this one (the sequential/"personalized" part), with
  (b) side features describing this interaction that are legitimately
      known beforehand: which skill it is, how many times the student has
      tried this skill, their prior success/failure count on it, the
      skill's difficulty, their rolling accuracy so far, and response-time
      features.
This also simplifies the output: instead of predicting a probability for
all 93 skills at every timestep (the baseline's DKT-style setup), the
model now predicts one probability per timestep for the skill actually
being attempted -- directly comparable to what the classical models do.

A HONEST NOTE ON THE 95% ACCURACY TARGET
--------------------------------------------
Worth being upfront: the strongest classical baseline (XGBoost, with the
same engineered features) reached 87.95% accuracy. Real student-response
data has an inherent noise ceiling — students make careless mistakes,
guess correctly on things they don't know, etc. — so 95% accuracy is an
ambitious target that may not be reachable on this dataset regardless of
tuning. This script is built to get as close as realistically possible
(feature augmentation + Optuna hyperparameter search + cross-validation +
regularization) rather than to fabricate a number — read the final
train/val/test accuracy gap it prints to judge whether it's over/underfit,
and treat "did we hit 95%" as an honest empirical question, not a
foregone conclusion.

OVERFITTING / UNDERFITTING CONTROL
--------------------------------------------
  - Dropout + L2 regularization are Optuna-tunable, so the search itself
    discovers how much regularization this dataset needs
  - K-fold cross-validation (not a single train/val split) is used as the
    Optuna objective, so a hyperparameter set can't just get lucky on one
    split
  - EarlyStopping (on validation loss, restore_best_weights=True) is used
    in every fold and in the final retrain
  - The FINAL held-out test set (68 students) is never touched during the
    Optuna search — it's only used once, at the very end, for the
    unbiased final number
"""

import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
)
import optuna

# !pip install optuna -q   # uncomment if optuna isn't already available in your Colab runtime

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
tf.get_logger().setLevel("ERROR")

# ------------------------------------------------------------------
# CONFIG — adjust these for speed vs. thoroughness
# ------------------------------------------------------------------
N_TRIALS = 30          # Optuna trials. Lower (e.g. 10-15) for a quick run.
N_FOLDS = 5             # cross-validation folds during the Optuna search
MAX_EPOCHS = 60         # upper bound; EarlyStopping will usually stop sooner
EARLY_STOP_PATIENCE = 8
BATCH_SIZE = 16

# ------------------------------------------------------------------
# STEP 1 — Load data and rebuild the SAME held-out test split used
# throughout this project, so results stay comparable
# ------------------------------------------------------------------
df = pd.read_csv("engineered_features.csv")
data = np.load("gru_sequences.npz")
skill_arr, correct_arr, interaction_arr, seq_lengths = (
    data["skill_arr"], data["correct_arr"], data["interaction_arr"], data["seq_lengths"]
)
num_skills = int(data["num_skills"])
max_len = skill_arr.shape[1]
sorted_user_ids = np.sort(df["user_id"].unique())
n_students = len(sorted_user_ids)

rng = np.random.default_rng(SEED)
students_for_split = df["user_id"].unique()
rng.shuffle(students_for_split)
n_test = int(len(students_for_split) * 0.2)
test_students = set(students_for_split[:n_test])
train_students = set(students_for_split[n_test:])

train_pool_idx = np.array([i for i, uid in enumerate(sorted_user_ids) if uid in train_students])
final_test_idx = np.array([i for i, uid in enumerate(sorted_user_ids) if uid in test_students])
print(f"Train pool (for CV + Optuna): {len(train_pool_idx)} students")
print(f"Held-out final test (untouched until the very end): {len(final_test_idx)} students")

# ------------------------------------------------------------------
# STEP 2 — Build base arrays shared across all folds/trials
# ------------------------------------------------------------------
START_TOKEN = 2 * num_skills + 1
PAD_TOKEN = 2 * num_skills
VOCAB_SIZE = 2 * num_skills + 2

# Shifted interaction input: at position t, the input is the ENCODING OF
# THE PREVIOUS interaction (skill+correctness at t-1); position 0 gets a
# dedicated START token since there's no history yet.
X_shifted_all = np.full((n_students, max_len), PAD_TOKEN, dtype=np.int32)
X_shifted_all[:, 0] = START_TOKEN
X_shifted_all[:, 1:] = interaction_arr[:, :-1]

valid_mask_all = np.zeros((n_students, max_len), dtype=np.float32)
for i, L in enumerate(seq_lengths):
    valid_mask_all[i, :L] = 1.0

y_all = np.where(correct_arr == -1, 0, correct_arr).astype(np.float32)  # -1 (pad) -> 0, masked out anyway

# Side features that do NOT need cross-student recomputation (each is
# purely a function of that one student's own past, already leakage-safe
# from the feature engineering step)
SIDE_FEATS_BASE = ["opportunity", "prior_success_count", "prior_failure_count",
                    "student_rolling_accuracy", "log_ms_first_response", "log_overlap_time"]

grouped = df.groupby("user_id")
per_student_rows = {uid: grouped.get_group(uid) for uid in sorted_user_ids}

base_side_all = np.zeros((n_students, max_len, len(SIDE_FEATS_BASE)), dtype=np.float32)
for i, uid in enumerate(sorted_user_ids):
    g = per_student_rows[uid]
    base_side_all[i, :len(g), :] = g[SIDE_FEATS_BASE].values


def build_skill_difficulty_column(train_idx_local, target_idx_local):
    """Recompute skill_difficulty using ONLY the given training students,
    then apply it to the target students (train or val/test). This avoids
    leaking any cross-student information the way a globally-fit
    skill_difficulty would."""
    train_uids = set(sorted_user_ids[i] for i in train_idx_local)
    train_rows = df[df["user_id"].isin(train_uids)]
    diff_map = train_rows.groupby("clean_skill_name")["correct"].mean()
    global_mean = train_rows["correct"].mean()

    col = np.zeros((len(target_idx_local), max_len, 1), dtype=np.float32)
    for j, i in enumerate(target_idx_local):
        uid = sorted_user_ids[i]
        g = per_student_rows[uid]
        vals = g["clean_skill_name"].map(diff_map).fillna(global_mean).values
        col[j, :len(g), 0] = vals
    return col


# ------------------------------------------------------------------
# STEP 3 — Model builder
# ------------------------------------------------------------------
def build_model(embed_dim, gru_units, dropout, l2reg, lr, n_side_feats):
    inter_in = keras.Input(shape=(max_len,), dtype="int32")
    side_in = keras.Input(shape=(max_len, n_side_feats), dtype="float32")
    emb = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=embed_dim)(inter_in)
    merged = layers.Concatenate()([emb, side_in])
    gru = layers.GRU(gru_units, return_sequences=True, dropout=dropout,
                      kernel_regularizer=keras.regularizers.l2(l2reg))(merged)
    out = layers.Dense(1, activation="sigmoid")(gru)   # shape (batch, T, 1) — keep 3D throughout
    model = keras.Model([inter_in, side_in], out)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy")
    return model


def masked_accuracy(y_true_2d, y_pred_2d, mask_2d):
    mask = mask_2d > 0
    return accuracy_score(y_true_2d[mask], (y_pred_2d[mask] >= 0.5).astype(int))


# ------------------------------------------------------------------
# STEP 4 — Optuna objective: K-fold CV over the train pool only
# ------------------------------------------------------------------
def objective(trial):
    embed_dim = trial.suggest_categorical("embed_dim", [16, 24, 32])
    gru_units = trial.suggest_categorical("gru_units", [32, 48, 64, 96])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    l2reg = trial.suggest_float("l2reg", 1e-6, 1e-2, log=True)
    lr = trial.suggest_float("lr", 3e-4, 5e-3, log=True)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_accs = []

    for tr_i, va_i in kf.split(train_pool_idx):
        tr_idx, va_idx = train_pool_idx[tr_i], train_pool_idx[va_i]

        diff_tr = build_skill_difficulty_column(tr_idx, tr_idx)
        diff_va = build_skill_difficulty_column(tr_idx, va_idx)
        side_tr = np.concatenate([base_side_all[tr_idx], diff_tr], axis=-1)
        side_va = np.concatenate([base_side_all[va_idx], diff_va], axis=-1)

        model = build_model(embed_dim, gru_units, dropout, l2reg, lr, side_tr.shape[-1])
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True
        )
        model.fit(
            [X_shifted_all[tr_idx], side_tr], y_all[tr_idx][..., None],
            sample_weight=valid_mask_all[tr_idx][..., None],
            validation_data=(
                [X_shifted_all[va_idx], side_va], y_all[va_idx][..., None],
                valid_mask_all[va_idx][..., None],
            ),
            epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop], verbose=0,
        )

        preds = model.predict([X_shifted_all[va_idx], side_va], verbose=0)[..., 0]
        acc = masked_accuracy(y_all[va_idx], preds, valid_mask_all[va_idx])
        fold_accs.append(acc)

        keras.backend.clear_session()  # free memory between folds

    mean_acc = float(np.mean(fold_accs))
    trial.set_user_attr("fold_accs", fold_accs)
    return mean_acc


print(f"\nStarting Optuna search: {N_TRIALS} trials x {N_FOLDS}-fold CV...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS)

print(f"\nBest CV accuracy: {study.best_value:.4f}")
print(f"Best hyperparameters: {study.best_params}")
print(f"Per-fold accuracies of best trial: {study.best_trial.user_attrs['fold_accs']}")

with open("optuna_study.pkl", "wb") as f:
    pickle.dump(study, f)

# ------------------------------------------------------------------
# STEP 5 — Retrain on the FULL train pool with the best hyperparameters,
# using a small internal split only to decide when to early-stop
# ------------------------------------------------------------------
best = study.best_params
rng2 = np.random.default_rng(SEED)
shuffled_pool = train_pool_idx.copy()
rng2.shuffle(shuffled_pool)
n_val = max(1, int(len(shuffled_pool) * 0.15))
final_val_idx = shuffled_pool[:n_val]
final_train_idx = shuffled_pool[n_val:]

diff_train_final = build_skill_difficulty_column(final_train_idx, final_train_idx)
diff_val_final = build_skill_difficulty_column(final_train_idx, final_val_idx)
diff_test_final = build_skill_difficulty_column(final_train_idx, final_test_idx)  # test uses TRAIN-derived difficulty only

side_train_final = np.concatenate([base_side_all[final_train_idx], diff_train_final], axis=-1)
side_val_final = np.concatenate([base_side_all[final_val_idx], diff_val_final], axis=-1)
side_test_final = np.concatenate([base_side_all[final_test_idx], diff_test_final], axis=-1)

final_model = build_model(best["embed_dim"], best["gru_units"], best["dropout"],
                           best["l2reg"], best["lr"], side_train_final.shape[-1])
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE,
                                             restore_best_weights=True)
history = final_model.fit(
    [X_shifted_all[final_train_idx], side_train_final], y_all[final_train_idx][..., None],
    sample_weight=valid_mask_all[final_train_idx][..., None],
    validation_data=(
        [X_shifted_all[final_val_idx], side_val_final], y_all[final_val_idx][..., None],
        valid_mask_all[final_val_idx][..., None],
    ),
    epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop], verbose=2,
)

final_model.save("best_gru_model.keras")

# ------------------------------------------------------------------
# STEP 6 — Final, ONE-TIME evaluation on the held-out test set
# ------------------------------------------------------------------
def evaluate(idx_set, side_feats, label):
    preds = final_model.predict([X_shifted_all[idx_set], side_feats], verbose=0)[..., 0]
    mask = valid_mask_all[idx_set] > 0
    y_t = y_all[idx_set][mask]
    y_p = preds[mask]
    y_pred_bin = (y_p >= 0.5).astype(int)

    acc = accuracy_score(y_t, y_pred_bin)
    prec = precision_score(y_t, y_pred_bin)
    rec = recall_score(y_t, y_pred_bin)
    f1 = f1_score(y_t, y_pred_bin)
    auc = roc_auc_score(y_t, y_p)
    cm = confusion_matrix(y_t, y_pred_bin)

    print(f"\n=== {label} ===")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | "
          f"F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
    print("Confusion Matrix:\n", cm)
    return {"accuracy": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "roc_auc": round(auc, 4), "confusion_matrix": cm.tolist()}


train_results = evaluate(final_train_idx, side_train_final, "Final TRAIN performance")
val_results = evaluate(final_val_idx, side_val_final, "Final VALIDATION performance")
test_results = evaluate(final_test_idx, side_test_final, "FINAL HELD-OUT TEST performance (the number that matters)")

gap = train_results["accuracy"] - test_results["accuracy"]
print(f"\nTrain-Test accuracy gap: {gap:.4f} "
      f"({'looks fine' if gap < 0.07 else 'some overfitting — consider more regularization'})")

if test_results["accuracy"] >= 0.95:
    print("\n>>> 95% accuracy target REACHED on the held-out test set.")
else:
    print(f"\n>>> Test accuracy is {test_results['accuracy']:.4f} — "
          f"below the 95% target. See the docstring at the top of this script "
          f"for why that ceiling may reflect real noise in the data rather than "
          f"an under-tuned model; compare against the train/val accuracy above "
          f"to judge whether more tuning vs. more/better data would help more.")

with open("final_gru_results.json", "w") as f:
    json.dump({
        "best_hyperparameters": best,
        "cv_accuracy": study.best_value,
        "train": train_results,
        "validation": val_results,
        "test": test_results,
        "train_test_accuracy_gap": round(gap, 4),
    }, f, indent=2)
print("\nSaved final_gru_results.json, best_gru_model.keras, optuna_study.pkl")