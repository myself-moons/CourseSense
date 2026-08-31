from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd


def plot_target_distribution(df: pd.DataFrame):
    target_counts = df["correct"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Incorrect (0)", "Correct (1)"], target_counts.values, color=["#e07a5f", "#3d9970"])
    ax.set_title("Target class distribution: Correct vs Incorrect")
    ax.set_ylabel("Number of interactions")
    fig.tight_layout()
    return fig


def plot_student_sequence_length_distribution(df: pd.DataFrame):
    seq_len = df.groupby("user_id").size()
    bins = [15, 25, 35, 45, 60, 80, 121]
    labels = ["15-24", "25-34", "35-44", "45-59", "60-79", "80-120"]
    bucket = pd.cut(seq_len, bins=bins, labels=labels, right=False, include_lowest=True)
    bucket_counts = bucket.value_counts().reindex(labels)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(bucket_counts.index, bucket_counts.values, color="#457b9d")
    ax.set_title("Student sequence length distribution")
    ax.set_xlabel("Interactions per student")
    ax.set_ylabel("Number of students")
    fig.tight_layout()
    return fig


def plot_top_skills(df: pd.DataFrame, n: int = 10):
    top_skills = df["clean_skill_name"].value_counts().head(n)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_skills.index[::-1], top_skills.values[::-1], color="#264653")
    ax.set_title(f"Top {n} most-practiced skills")
    ax.set_xlabel("Number of attempts")
    fig.tight_layout()
    return fig


def plot_skill_difficulty(df: pd.DataFrame, min_attempts: int = 100):
    skill_counts = df["clean_skill_name"].value_counts()
    frequent_skills = skill_counts[skill_counts >= min_attempts].index
    skill_acc = df[df["clean_skill_name"].isin(frequent_skills)].groupby("clean_skill_name")["correct"].mean().sort_values()
    hardest_easiest = pd.concat([skill_acc.head(5), skill_acc.tail(5)])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(hardest_easiest.index, hardest_easiest.values, color="#e76f51")
    ax.set_title(f"Hardest vs. easiest skills (min. {min_attempts} attempts)")
    ax.set_xlabel("Correctness rate")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig


def plot_learning_curve(df: pd.DataFrame):
    df_sorted = df.sort_values(["user_id", "clean_skill_name", "order_id"]).copy()
    df_sorted["opportunity"] = df_sorted.groupby(["user_id", "clean_skill_name"]).cumcount() + 1
    lc = df_sorted[df_sorted["opportunity"] <= 10].groupby("opportunity")["correct"].agg(["mean", "count"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lc.index, lc["mean"], marker="o", color="#2a9d8f")
    ax.set_title("Learning curve: accuracy by practice attempt on a skill")
    ax.set_xlabel("Opportunity number (nth time practicing this skill)")
    ax.set_ylabel("Correctness rate")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_attempt_and_hint_usage(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if "attempt_count" in df.columns:
        attempt_counts = df["attempt_count"].value_counts().sort_index().head(10)
        axes[0].bar(attempt_counts.index.astype(str), attempt_counts.values, color="#2a9d8f")
        axes[0].set_title("Attempt count distribution")
        axes[0].set_xlabel("Attempt count")
        axes[0].set_ylabel("Interactions")

    if "hint_count" in df.columns:
        hint_result = df.groupby(df["hint_count"] > 0)["correct"].mean()
        labels = ["No hint", "Hint used"]
        axes[1].bar(labels, [hint_result.get(False, 0), hint_result.get(True, 0)], color=["#4cc9f0", "#ef476f"])
        axes[1].set_title("Correctness rate by hint usage")
        axes[1].set_ylabel("Correctness rate")
        axes[1].set_ylim(0, 1)

    fig.tight_layout()
    return fig
