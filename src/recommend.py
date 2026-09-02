from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

MIN_SKILL_OCCURRENCES = 20
PAD_TOKEN = 186
START_TOKEN = 187
MAX_LEN = 120
FEATURE_COLUMNS = [
    "opportunity",
    "prior_success_count",
    "prior_failure_count",
    "student_rolling_accuracy",
    "log_ms_first_response",
    "log_overlap_time",
    "skill_difficulty",
]


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_engineered_features() -> pd.DataFrame:
    return pd.read_csv(_root_dir() / "data" / "engineered_features.csv")


def _load_sequence_data() -> Dict[str, np.ndarray]:
    with np.load(_root_dir() / "data" / "gru_sequences.npz") as data:
        return {key: data[key] for key in data.files}


def _load_skill_vocab() -> Dict[str, int]:
    with open(_root_dir() / "data" / "skill_vocab.json") as f:
        return json.load(f)


def _sorted_student_ids(features_df: pd.DataFrame) -> np.ndarray:
    return np.sort(features_df["user_id"].dropna().unique())


def _resolve_student_index(student_row_idx_or_user_id, sorted_student_ids: np.ndarray) -> int:
    if isinstance(student_row_idx_or_user_id, (int, np.integer)):
        candidate = int(student_row_idx_or_user_id)
        if 0 <= candidate < len(sorted_student_ids):
            return candidate
        if candidate in sorted_student_ids.astype(int).tolist():
            return int(np.where(sorted_student_ids == candidate)[0][0])
        raise IndexError("Student row index out of range.")

    user_id = int(student_row_idx_or_user_id)
    matches = np.where(sorted_student_ids == user_id)[0]
    if len(matches) == 0:
        raise KeyError(f"Student user_id {user_id} was not found.")
    return int(matches[0])


def _student_history_for_index(student_idx: int, sequence_data: Dict[str, np.ndarray]):
    seq_len = int(sequence_data["seq_lengths"][student_idx])
    return {
        "seq_len": seq_len,
        "skill": sequence_data["skill_arr"][student_idx, :seq_len],
        "correct": sequence_data["correct_arr"][student_idx, :seq_len],
        "interaction": sequence_data["interaction_arr"][student_idx, :seq_len],
    }


def _summarize_student_skill_history(skill_history: np.ndarray, correct_history: np.ndarray) -> Dict[int, Dict[str, int]]:
    """Keys here are skill_idx (0-92, the SAME space as skill_arr/skill_vocab.json) —
    this must stay consistent with whatever ID space candidate skills are looked up in."""
    stats: Dict[int, Dict[str, int]] = {}
    for skill_id, is_correct in zip(skill_history.tolist(), correct_history.tolist()):
        if int(skill_id) < 0:
            continue
        key = int(skill_id)
        bucket = stats.setdefault(key, {"opportunity": 0, "success": 0, "failure": 0})
        bucket["opportunity"] += 1
        if int(is_correct) == 1:
            bucket["success"] += 1
        else:
            bucket["failure"] += 1
    return stats


def _candidate_skill_row(
    skill_idx: int,
    student_skill_stats: Dict[int, Dict[str, int]],
    student_rows: pd.DataFrame,
    skill_difficulty_lookup: pd.Series,
) -> np.ndarray:
    prior_success = int(student_skill_stats.get(skill_idx, {}).get("success", 0))
    prior_failure = int(student_skill_stats.get(skill_idx, {}).get("failure", 0))
    opportunity = int(student_skill_stats.get(skill_idx, {}).get("opportunity", 0)) + 1

    student_rolling_accuracy = float(student_rows["student_rolling_accuracy"].iloc[-1]) if not student_rows.empty else 0.0
    mean_log_ms = float(student_rows["log_ms_first_response"].mean()) if not student_rows.empty else 0.0
    mean_log_overlap = float(student_rows["log_overlap_time"].mean()) if not student_rows.empty else 0.0
    skill_difficulty = float(skill_difficulty_lookup.get(skill_idx, 0.0))

    return np.array(
        [
            float(opportunity),
            float(prior_success),
            float(prior_failure),
            float(student_rolling_accuracy),
            float(mean_log_ms),
            float(mean_log_overlap),
            float(skill_difficulty),
        ],
        dtype=np.float32,
    )


def _build_hypothetical_sequence(
    student_idx: int,
    skill_idx: int,
    features_df: pd.DataFrame,
    sequence_data: Dict[str, np.ndarray],
    student_skill_stats: Dict[int, Dict[str, int]],
    skill_difficulty_lookup: pd.Series,
    sorted_student_ids: np.ndarray,
):
    seq_len = int(sequence_data["seq_lengths"][student_idx])
    student_user_id = int(sorted_student_ids[student_idx])
    student_rows = features_df[features_df["user_id"] == student_user_id].sort_values("order_id").reset_index(drop=True)

    interaction_seq = np.full((MAX_LEN,), PAD_TOKEN, dtype=np.int32)
    side_features = np.zeros((MAX_LEN, len(FEATURE_COLUMNS)), dtype=np.float32)

    if seq_len > 0:
        interaction_seq[:seq_len] = sequence_data["interaction_arr"][student_idx, :seq_len]
        historical_side = student_rows[FEATURE_COLUMNS].iloc[:seq_len].to_numpy(dtype=np.float32)
        if historical_side.shape[0] > 0:
            side_features[:seq_len] = historical_side
        interaction_seq[seq_len] = int(sequence_data["interaction_arr"][student_idx, seq_len - 1])
    else:
        interaction_seq[0] = START_TOKEN

    side_features[min(seq_len, MAX_LEN - 1)] = _candidate_skill_row(skill_idx, student_skill_stats, student_rows, skill_difficulty_lookup)
    return interaction_seq, side_features


def _sort_candidate_skills(candidate_records: List[Dict[str, Any]], strategy: str) -> List[Dict[str, Any]]:
    strategy_key = (strategy or "").lower().strip()
    if strategy_key == "zpd":
        return sorted(candidate_records, key=lambda item: abs(float(item["predicted_success_prob"]) - 0.60))
    return sorted(candidate_records, key=lambda item: float(item["predicted_success_prob"]))


def recommend_next_skills(student_row_idx_or_user_id, top_k, strategy):
    features_df = _load_engineered_features()
    sequence_data = _load_sequence_data()
    skill_to_idx = _load_skill_vocab()          # clean_skill_name -> 0..92 (the model's actual vocabulary)
    idx_to_skill = {v: k for k, v in skill_to_idx.items()}

    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(str(_root_dir() / "models" / "best_gru_model.keras"))
    except Exception as exc:  # pragma: no cover - handled in app UI via graceful warning
        raise RuntimeError("The GRU model could not be loaded.") from exc

    sorted_student_ids = _sorted_student_ids(features_df)
    student_idx = _resolve_student_index(student_row_idx_or_user_id, sorted_student_ids)
    history = _student_history_for_index(student_idx, sequence_data)
    # skill_arr already stores skill_idx (0-92) — this is consistent with skill_vocab.json by construction
    student_skill_stats = _summarize_student_skill_history(history["skill"], history["correct"])

    # Candidate pool and occurrence filtering, all in the SAME 0-92 space as the model
    features_df["_skill_idx"] = features_df["clean_skill_name"].map(skill_to_idx)
    skill_counts = features_df["_skill_idx"].value_counts().sort_index()
    candidate_skill_indices = [int(s) for s, count in skill_counts.items() if int(count) >= MIN_SKILL_OCCURRENCES]
    excluded_skill_indices = [int(s) for s, count in skill_counts.items() if int(count) < MIN_SKILL_OCCURRENCES]
    excluded_skill_names = [idx_to_skill[s] for s in excluded_skill_indices]

    if not candidate_skill_indices:
        empty_df = pd.DataFrame(columns=["rank", "skill_idx", "skill_name", "predicted_success_prob", "opportunity", "prior_success_count", "prior_failure_count"])
        return empty_df, excluded_skill_names

    # skill_difficulty already lives in engineered_features.csv, keyed by clean_skill_name/_skill_idx
    skill_difficulty_lookup = features_df.groupby("_skill_idx")["skill_difficulty"].mean()

    batched_interactions = []
    batched_side_features = []
    candidate_records = []

    for skill_idx in candidate_skill_indices:
        interaction_seq, side_features = _build_hypothetical_sequence(
            student_idx=student_idx,
            skill_idx=skill_idx,
            features_df=features_df,
            sequence_data=sequence_data,
            student_skill_stats=student_skill_stats,
            skill_difficulty_lookup=skill_difficulty_lookup,
            sorted_student_ids=sorted_student_ids,
        )
        batched_interactions.append(interaction_seq)
        batched_side_features.append(side_features)
        candidate_records.append({"skill_idx": int(skill_idx)})

    batched_interactions = np.stack(batched_interactions, axis=0)
    batched_side_features = np.stack(batched_side_features, axis=0)

    input_batches = [batched_interactions, batched_side_features]
    model_input_names = [input_tensor.name for input_tensor in model.inputs]
    prediction_inputs = {name: batch for name, batch in zip(model_input_names, input_batches)}
    predictions = model.predict(prediction_inputs, verbose=0)

    if predictions.ndim == 3:
        probability_index = min(int(history["seq_len"]), MAX_LEN - 1)
        probs = predictions[:, probability_index, 0]
    elif predictions.ndim == 2:
        probs = predictions[:, 0]
    else:
        probs = np.asarray(predictions).reshape(-1)

    for idx, skill_idx in enumerate(candidate_skill_indices):
        candidate_records[idx].update(
            {
                "skill_name": idx_to_skill[skill_idx],
                "predicted_success_prob": float(probs[idx]),
                "opportunity": int(student_skill_stats.get(skill_idx, {}).get("opportunity", 0)) + 1,
                "prior_success_count": int(student_skill_stats.get(skill_idx, {}).get("success", 0)),
                "prior_failure_count": int(student_skill_stats.get(skill_idx, {}).get("failure", 0)),
            }
        )

    ordered = _sort_candidate_skills(candidate_records, strategy)
    top_candidates = ordered[: int(top_k)]

    result_df = pd.DataFrame(top_candidates)
    if not result_df.empty:
        result_df = result_df[["skill_idx", "skill_name", "predicted_success_prob", "opportunity", "prior_success_count", "prior_failure_count"]].copy()
        result_df.insert(0, "rank", range(1, len(result_df) + 1))
        result_df["predicted_success_prob"] = result_df["predicted_success_prob"].round(4)

    return result_df, excluded_skill_names