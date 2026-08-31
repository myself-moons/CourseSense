from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def plot_correct_incorrect_balance(df: pd.DataFrame):
    counts = df["correct"].value_counts().reindex([0, 1]).fillna(0).astype(int)
    labels = ["Incorrect", "Correct"]
    values = counts.to_list()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, values, color=["#ef476f", "#06d6a0"])
    ax.set_title("Correct vs. Incorrect Responses")
    ax.set_ylabel("Interactions")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_interactions_per_student(df: pd.DataFrame):
    counts = df["user_id"].value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(counts.values, bins=20, color="#4cc9f0", edgecolor="black")
    ax.set_title("Distribution of Interactions per Student")
    ax.set_xlabel("Interactions per student")
    ax.set_ylabel("Number of students")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_top_skills(df: pd.DataFrame, n: int = 15):
    skill_counts = (
        df.assign(skill_id=pd.to_numeric(df["skill_id"], errors="coerce"))
        .groupby("skill_id", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
        .head(n)
    )
    skill_names = []
    for skill_id in skill_counts["skill_id"].tolist():
        name = df.loc[df["skill_id"].astype(str) == str(skill_id), "skill_name"].iloc[0]
        skill_names.append(name)
    skill_counts["skill_name"] = skill_names

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(skill_counts["skill_name"], skill_counts["count"], color="#5e60ce")
    ax.invert_yaxis()
    ax.set_title(f"Top {n} Skills by Frequency")
    ax.set_xlabel("Occurrences")
    ax.set_ylabel("Skill")
    fig.tight_layout()
    return fig


def plot_skill_difficulty_spread(df: pd.DataFrame):
    skill_stats = (
        df.assign(skill_id=pd.to_numeric(df["skill_id"], errors="coerce"))
        .groupby("skill_id")
        .agg(skill_name=("skill_name", "first"), count=("skill_id", "size"), avg_difficulty=("skill_difficulty", "mean"))
        .reset_index()
    )
    filtered = skill_stats[skill_stats["count"] >= 20].copy()
    filtered = filtered.sort_values("avg_difficulty", ascending=False)

    top_hardest = filtered.head(10)
    top_easiest = filtered.sort_values("avg_difficulty", ascending=True).head(10)
    combined = pd.concat([top_hardest, top_easiest], ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#ef476f" if v >= filtered["avg_difficulty"].median() else "#8ecae6" for v in combined["avg_difficulty"]]
    ax.bar(combined["skill_name"], combined["avg_difficulty"], color=colors)
    ax.set_title("Hardest vs Easiest Skills (>= 20 Occurrences)")
    ax.set_xlabel("Skill")
    ax.set_ylabel("Mean skill difficulty")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_metric_comparison(df_metrics: pd.DataFrame):
    if df_metrics.empty:
        return None

    metric_cols = [col for col in ["accuracy", "precision", "recall", "f1", "roc_auc"] if col in df_metrics.columns]
    if not metric_cols:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    for metric in metric_cols:
        ax.plot(df_metrics["split"], df_metrics[metric], marker="o", label=metric)
    ax.set_title("Model Metrics by Split")
    ax.set_ylabel("Score")
    ax.set_xlabel("Split")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_optuna_history(study):
    if study is None:
        return None

    trials = getattr(study, "trials", None)
    if trials is None:
        return None

    values = []
    for trial in trials:
        if hasattr(trial, "value") and trial.value is not None:
            values.append((trial.number, float(trial.value)))
    if not values:
        return None

    numbers, objective_values = zip(*values)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(numbers, objective_values, marker="o", linewidth=1.5)
    ax.set_title("Optuna optimization history")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Objective value")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig
