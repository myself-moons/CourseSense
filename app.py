from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.charts import (
    plot_correct_incorrect_balance,
    plot_interactions_per_student,
    plot_skill_difficulty_spread,
    plot_top_skills,
)
from src.eda_utils import (
    plot_attempt_and_hint_usage,
    plot_learning_curve,
    plot_skill_difficulty,
    plot_student_sequence_length_distribution,
    plot_target_distribution,
    plot_top_skills as plot_eda_top_skills,
)
from src.recommend import MIN_SKILL_OCCURRENCES, recommend_next_skills


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


@st.cache_resource(show_spinner=False)
def load_model():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - surfaced in app UI instead of crashing
        raise RuntimeError("TensorFlow is not installed. Run pip install -r requirements.txt first.") from exc

    model_path = MODEL_DIR / "best_gru_model.keras"
    return tf.keras.models.load_model(str(model_path))


@st.cache_data(show_spinner=False)
def load_engineered_features() -> pd.DataFrame:
    csv_path = DATA_DIR / "engineered_features.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected dataset at {csv_path}, but it is missing.")
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner=False)
def load_sequence_data() -> Dict[str, np.ndarray]:
    npz_path = DATA_DIR / "gru_sequences.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Expected sequence file at {npz_path}, but it is missing.")
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


@st.cache_data(show_spinner=False)
def load_results_json() -> Dict[str, Any]:
    json_path = DATA_DIR / "final_gru_results.json"
    if not json_path.exists():
        return {}
    return pd.read_json(json_path).to_dict(orient="records") if json_path.exists() else {}


@st.cache_data(show_spinner=False)
def load_results() -> Dict[str, Any]:
    json_path = DATA_DIR / "final_gru_results.json"
    if not json_path.exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as handle:
        return __import__("json").load(handle)


@st.cache_data(show_spinner=False)
def load_baseline_results() -> Dict[str, Any]:
    json_path = DATA_DIR / "baseline_results.json"
    if not json_path.exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as handle:
        return __import__("json").load(handle)


@st.cache_data(show_spinner=False)
def load_optuna_study():
    study_path = DATA_DIR / "optuna_study.pkl"
    if not study_path.exists():
        return None
    try:
        import pickle

        with open(study_path, "rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


st.set_page_config(page_title="CourseSense", page_icon="🎓", layout="wide")


def render_chart_with_explainer(title: str, fig, explanation: str):
    st.subheader(title)
    if fig is not None:
        fig.tight_layout()
        left, mid, right = st.columns([1, 3, 1])
        with mid:
            st.pyplot(fig)
    with st.expander("Chart Information", expanded=False):
        st.write(explanation)


def render_overview_tab():
    st.title("CourseSense")
    st.write(
        "CourseSense predicts whether a student will answer a math skill correctly using their recent interaction history. "
        "A GRU model processes the student’s sequence of past (skill, correctness) events, estimates a per-step success probability, "
        "and those probabilities feed a personalized recommendation engine for the next skill to practice."
    )

    metrics_df = load_engineered_features()
    student_count = int(metrics_df["user_id"].nunique()) if not metrics_df.empty else 0
    skill_count = int(metrics_df["skill_id"].nunique()) if not metrics_df.empty else 0
    interaction_count = int(len(metrics_df))

    results = load_results()
    test_accuracy = results.get("test", {}).get("accuracy") if isinstance(results, dict) else None
    test_roc_auc = results.get("test", {}).get("roc_auc") if isinstance(results, dict) else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Students", f"{student_count:,.0f}")
    col2.metric("Skills", f"{skill_count:,.0f}")
    col3.metric("Interactions", f"{interaction_count:,.0f}")
    col4.metric("Test accuracy", f"{test_accuracy:.3f}" if test_accuracy is not None else "N/A")
    st.metric("Test ROC-AUC", f"{test_roc_auc:.3f}" if test_roc_auc is not None else "N/A")

    st.subheader("AFTER dataset preview")
    after_path = DATA_DIR / "assistments_sample_AFTER.csv"
    if after_path.exists():
        preview_df = pd.read_csv(after_path).head(8)
        st.dataframe(preview_df, use_container_width=False)
        st.caption("This is a preview of the cleaned AFTER dataset used for the EDA and skill-analysis steps.")
    else:
        st.info("The AFTER dataset preview is not available in the data folder yet.")


def render_dataset_explorer_tab():
    st.header("Dataset Explorer")
    df = load_engineered_features()
    if df.empty:
        st.warning("The engineered_features.csv file is not available in the data directory.")
        return

    render_chart_with_explainer(
        "Correct vs incorrect responses",
        plot_correct_incorrect_balance(df),
        "This chart compares how often students answered a question correctly versus incorrectly. A larger green bar means students were usually successful on the item, while a larger pink bar means the item was harder and more often answered incorrectly.",
    )

    render_chart_with_explainer(
        "Distribution of interactions per student",
        plot_interactions_per_student(df),
        "This shows how many learning events each student has in the dataset. A long tail means a few students have many more interactions than others, which is common in real education data and can affect how the model learns.",
    )

    render_chart_with_explainer(
        "Top 15 skills by frequency",
        plot_top_skills(df, n=15),
        "This highlights the skills students practice most often. It helps us see which topics are most common in the data and which ones may need more attention in the recommendation system.",
    )

    render_chart_with_explainer(
        "Skill difficulty spread",
        plot_skill_difficulty_spread(df),
        "This chart shows the range of difficulty across skills. Skills near the left side are easier for students to answer correctly, while skills toward the right are more challenging and may need more practice support.",
    )


def render_eda_results_tab():
    st.header("EDA Results")
    dataset_path = DATA_DIR / "assistments_sample_AFTER.csv"
    if not dataset_path.exists():
        st.warning("The EDA dataset file assistments_sample_AFTER.csv is not available in data/, so the EDA charts cannot be rendered.")
        return

    df = pd.read_csv(dataset_path)

    render_chart_with_explainer(
        "Target distribution",
        plot_target_distribution(df),
        "This chart tells us whether the dataset is balanced between correct and incorrect answers. A balanced dataset is easier for a model to learn from, while a strongly skewed dataset can make predictions less reliable for the minority outcome.",
    )

    render_chart_with_explainer(
        "Student sequence length distribution",
        plot_student_sequence_length_distribution(df),
        "This summarizes how many actions each student has in their learning history. Longer sequences usually give the model more context to predict future performance, but very short sequences can make learning harder.",
    )

    render_chart_with_explainer(
        "Top skills",
        plot_eda_top_skills(df, n=10),
        "This shows which knowledge skills appear most often in the data. It helps explain which topics students spend the most time on and where the dataset is strongest for pattern learning.",
    )

    render_chart_with_explainer(
        "Skill difficulty",
        plot_skill_difficulty(df, min_attempts=100),
        "This compares the average success rate of different skills. Skills with lower correctness rates are usually harder, while those with higher rates are easier for students to master.",
    )

    render_chart_with_explainer(
        "Learning curve",
        plot_learning_curve(df),
        "This shows how accuracy changes as a student practices the same skill multiple times. In a healthy learning trend, performance should rise over repeated practice, which makes the model more useful for recommendation.",
    )

    render_chart_with_explainer(
        "Attempt and hint usage",
        plot_attempt_and_hint_usage(df),
        "These charts show whether students are making repeated attempts and whether hints are associated with better or worse outcomes. This helps us understand student behavior and whether hints are signaling difficulty or support.",
    )


def _safe_load_results() -> Dict[str, Any]:
    result_obj = load_results()
    if not isinstance(result_obj, dict) or not result_obj:
        return {}
    return result_obj


def render_model_performance_tab():
    st.header("Model Performance")
    results = _safe_load_results()
    baseline_results = load_baseline_results()

    if not results:
        st.warning("No model metrics file was found in data/final_gru_results.json. The app will show the structure here once the file is present.")
        return

    split_order = ["train", "val", "test"]
    metric_rows = []
    for split_name in split_order:
        split_metrics = results.get(split_name, {})
        if not split_metrics:
            continue
        metric_rows.append(
            {
                "Split": split_name,
                "accuracy": split_metrics.get("accuracy"),
                "precision": split_metrics.get("precision"),
                "recall": split_metrics.get("recall"),
                "f1": split_metrics.get("f1"),
                "roc_auc": split_metrics.get("roc_auc"),
            }
        )

    metric_df = pd.DataFrame(metric_rows)
    if not metric_df.empty:
        st.dataframe(metric_df, use_container_width=True)

    st.subheader("Confusion matrices")
    for split_name in split_order:
        confusion = results.get(split_name, {}).get("confusion_matrix")
        if confusion is None:
            continue
        st.markdown(f"### {split_name.upper()} split")
        heatmap_data = np.asarray(confusion, dtype=int)
        st.write(pd.DataFrame(heatmap_data, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        heatmap = ax.imshow(heatmap_data, cmap="Blues")
        ax.set_xticks(np.arange(2), labels=["Pred 0", "Pred 1"])
        ax.set_yticks(np.arange(2), labels=["Actual 0", "Actual 1"])
        for row in range(heatmap_data.shape[0]):
            for col in range(heatmap_data.shape[1]):
                value = heatmap_data[row, col]
                ax.text(col, row, str(value), ha="center", va="center", color="black" if value <= heatmap_data.max() / 2 else "white")
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{split_name.upper()} confusion matrix")
        fig.tight_layout()
        with st.expander(" confusion matrix info", expanded=False):
            st.write(
                "A confusion matrix compares the model's prediction with the true label. The top-left and bottom-right cells are the correct predictions, while the off-diagonal cells show the mistakes. If most values fall on the diagonal, the model is making mostly correct recommendations."
            )
        left, mid, right = st.columns([1, 6, 1])
        with mid:
            st.pyplot(fig)

    st.subheader("Best hyperparameters")
    best_params = results.get("best_hyperparameters")
    if best_params:
        st.json(best_params)
    else:
        st.info("No best_hyperparameters field was found in the results JSON.")

    if baseline_results:
        st.subheader("Baseline model comparison")
        baseline_map = baseline_results.get("results", {})
        if isinstance(baseline_map, dict) and baseline_map:
            baseline_df = pd.DataFrame(
                [
                    {
                        "Model": model_name,
                        "Accuracy": metrics.get("accuracy"),
                        "Precision": metrics.get("precision"),
                        "Recall": metrics.get("recall"),
                        "F1": metrics.get("f1"),
                        "ROC-AUC": metrics.get("roc_auc"),
                    }
                    for model_name, metrics in baseline_map.items()
                ]
            ).sort_values("Accuracy", ascending=False)
            st.dataframe(baseline_df, use_container_width=True)

            chart_df = baseline_df.set_index("Model")[["Accuracy", "F1", "ROC-AUC"]].copy()
            left, mid, right = st.columns([1, 6, 1])
            with mid:
                st.line_chart(chart_df)
            with st.expander("Baseline comparison Info", expanded=False):
                st.write(
                    "This compares several simple baseline models against each other using the same task. Higher bars or values mean stronger performance on average, and it helps us understand whether the GRU model is improving beyond simpler approaches."
                )

    st.subheader("Optimization history")
    try:
        study = load_optuna_study()
        if study is None:
            st.info("No Optuna study pickle was found, or it failed to deserialize.")
        else:
            trials = getattr(study, "trials", [])
            if not trials:
                st.info("The Optuna study file was loaded but contains no recorded trials.")
            else:
                history = [{"trial_number": trial.number, "objective_value": float(trial.value)} for trial in trials if getattr(trial, "value", None) is not None]
                if history:
                    history_df = pd.DataFrame(history)
                    left, mid, right = st.columns([1, 6, 1])
                    with mid:
                        st.line_chart(history_df.set_index("trial_number")["objective_value"])
                    with st.expander("What does the optimization history show?", expanded=False):
                        st.write(
                            "This chart tracks how the model's objective value changed as hyperparameters were tuned."
                        )
                else:
                    st.info("The Optuna study file was loaded but no objective values were recorded.")
    except Exception:
        st.warning("The Optuna study file could not be loaded. This is usually caused by a version mismatch or an incompatible pickle.")

    st.caption("Classical ML baseline comparison (Naive Bayes, Logistic Regression, Random Forest, XGBoost) is shown above using the project’s live baseline_results.json values.")


def render_recommender_tab():
    st.header("Try the Recommender")
    df = load_engineered_features()
    if df.empty:
        st.warning("The engineered_features.csv file is missing, so the recommendation tab cannot run.")
        return

    model = load_model()
    sequence_data = load_sequence_data()

    sorted_user_ids = np.sort(df["user_id"].dropna().unique())
    selected_user = st.selectbox("Choose a student by user_id", sorted_user_ids.astype(int).tolist())

    strategy = st.radio(
        "Recommendation strategy",
        ["Remediate (weakest skills)", "ZPD (challenging but achievable, ~60% predicted success)"],
        index=0,
    )
    with st.expander("What do these modes mean?", expanded=False):
        st.write(
            "- Remediation mode: choose the skill the student is currently weakest at. This is useful for practice that targets gaps. In this mode, a lower predicted success rate can rank higher because the system is intentionally looking for the hardest area to improve.\n"
            "- ZPD mode: choose a skill that is challenging but still realistic for the student, around a 60% success rate. This is a stretch goal: not too easy, not too hard."
        )
    top_k = st.slider("Number of recommendations", min_value=3, max_value=10, value=5, step=1)

    run_recommendation = st.button("Run recommendation")

    if run_recommendation:
        strategy_key = "remediate" if strategy.startswith("Remediate") else "zpd"
        results, excluded_skills = recommend_next_skills(
            student_row_idx_or_user_id=int(selected_user),
            top_k=int(top_k),
            strategy=strategy_key,
        )

        if results is None or results.empty:
            st.info("No skills met the minimum occurrence threshold for a recommendation.")
        else:
            st.subheader("Recommended skills")
            st.dataframe(results, use_container_width=False)

            top_skill = results.iloc[0]
            top_prob = float(top_skill["predicted_success_prob"])
            st.success(
                f"The best next skill for this student is {top_skill['skill_name']} with a predicted success rate of {top_prob:.1%}. "
                "This is the top recommendation in the selected strategy, not necessarily the highest-scoring skill overall. In Remediation mode, the model is purposely ranking the weakest areas higher so the student can practice and improve them."
            )

            with st.expander("What do these model scores mean?", expanded=False):
                st.write(
                    "Each value is a probability between 0 and 1. A score near 1.0 means the model believes the student is very likely to answer that skill correctly. "
                    "A score near 0.0 means it expects a struggle or a wrong answer. In Remediation mode, lower scores can rank higher because the goal is to target a skill the student is currently struggling with. In ZPD mode, the goal is different: pick a skill around about 60% success, which is challenging but still achievable."
                )

            st.bar_chart(results.set_index("skill_name")["predicted_success_prob"].sort_values())

        if excluded_skills:
            with st.expander("Insufficient data for these skills"):
                st.write(", ".join(excluded_skills))


def render_model_working_tab():
    st.header("Model Working")
    st.markdown(
        "This section explains the end-to-end workflow used by the GRU model in the project files [src/gru_training.py](src/gru_training.py) and [src/gru_recommender.py](src/gru_recommender.py). The goal is to keep it simple: the model learns from a student’s recent activity and then predicts which skill is the best next practice target."
    )

    st.subheader("1) Data preparation and preprocessing")
    st.markdown(
        "The project starts with student interaction data in the CSV and sequence files. Before modeling, the preprocessing script in [src/preproessing.py](src/preproessing.py) downloads the raw ASSISTments data, samples it down to a manageable student-sequence dataset, and keeps each student’s full interaction history intact.\n\n"
        "This stage does important cleanup work:\n"
        "- keeps only the columns needed for sequential modeling\n"
        "- samples whole students rather than random rows, so the GRU sees realistic timelines\n"
        "- saves a BEFORE file with the raw sampled dataset\n"
        "- cleans the `skill_name` text by lowercasing, removing punctuation, removing stop words, and lemmatizing the words\n"
        "- saves an AFTER file with a new `clean_skill_name` field for downstream analysis\n\n"
        "From this, the system builds:\n"
        "- a sequence of skills the student has seen\n"
        "- whether each answer was correct or incorrect\n"
        "- features such as prior success counts, prior failure counts, rolling accuracy, skill difficulty, and response times\n\n"
        "These are the signals the model uses to understand learning progress."
    )

    st.subheader("2) Training the GRU model")
    st.markdown(
        "The model in [src/gru_training.py](src/gru_training.py) is a GRU, which is a type of neural network designed for sequence data. It is good at learning patterns over time, such as: \"this student usually does well on skill A after practicing skill B several times.\"\n\n"
        "The input is not just the skill ID. It also includes side features such as:\n"
        "- how often the student has seen the skill\n"
        "- how many times they succeeded or failed on it\n"
        "- their overall recent accuracy\n"
        "- the difficulty of that skill\n\n"
        "This helps the model predict whether the next answer will be correct, using both the student’s history and the current skill context."
    )

    st.subheader("3) What the model predicts")
    st.markdown(
        "At each step, the model tries to answer a simple question:\n\n"
        "\"Given what this student has done so far, what is the chance they get the next attempt correct?\"\n\n"
        "The output is a probability between 0 and 1. A value near 1 means the model thinks the student is likely to succeed. A value near 0 means it thinks the student is likely to struggle."
    )

    st.subheader("4) Building a recommendation")
    st.markdown(
        "The recommendation logic in [src/gru_recommender.py](src/gru_recommender.py) does a what-if check for each candidate skill. It takes the student’s actual history and creates a hypothetical next step for every possible skill. Then it asks the trained model: \"If this student tries skill X next, what is the chance they get it right?\"\n\n"
        "This is done for many skills at once, so the system can compare them fairly."
    )

    st.subheader("5) Ranking the skills")
    st.markdown(
        "After the model scores all candidate skills, the app ranks them according to the selected strategy:\n\n"
        "- Remediation: choose the skills with the lowest predicted success probability. This means the student is likely struggling most there, so the app suggests a skill to strengthen.\n"
        "- ZPD: choose skills close to a 60% predicted success rate. This is a challenging-but-achievable zone, where the task is hard enough to learn but not so hard that it feels impossible.\n\n"
        "This is why a skill with a lower probability can rank above a higher one in Remediation mode: the goal is not to pick the easiest skill, but to focus on the weakest area."
    )

    st.subheader("6) Final recommendation shown to the user")
    st.markdown(
        "The app finally shows the top few skills in a table and a message like: \"The best next skill for this student is X with a predicted success rate of 51.9%.\"\n\n"
        "That does not mean the model thinks that skill is the easiest. It means the model thinks it is the best next target for the selected strategy."
    )

    st.info(
        "In short: the model learns from a student’s history, predicts success for each possible next skill, and ranks the skills based on the chosen teaching goal: fix weak gaps or choose a stretch challenge."
    )


def render_conclusion_tab():
    st.header("Conclusion")
    results = _safe_load_results()

    if results:
        test_accuracy = results.get("test", {}).get("accuracy")
        test_roc_auc = results.get("test", {}).get("roc_auc")
        if test_accuracy is not None and test_roc_auc is not None:
            st.write(
                f"The live test result is {test_accuracy:.3f} accuracy and {test_roc_auc:.3f} ROC-AUC. "
                "In plain English, the model is correctly predicting the next answer around 88% of the time on fresh data, which is strong for educational prediction but still below the ideal 95% target. "
                "This is a realistic result for student learning data because the signals are noisy, students vary a lot, and some skills are much harder than others."
            )
        else:
            st.write(
                "The project goal was to reach around 95% accuracy, but the achieved result is lower than this target; the exact live values will appear here once the metrics file is present in the data folder."
            )

        st.subheader("What the numbers mean")
        st.markdown(
            f"- Accuracy: {test_accuracy:.3f} means the model got the answer right about {test_accuracy * 100:.1f}% of the time on unseen test data.\n"
            f"- ROC-AUC: {test_roc_auc:.3f} means the model is good at separating easier questions from harder ones and ranking likely correct answers above likely incorrect ones.\n"
            "- This is not a perfect classroom grade; it is a practical prediction tool that helps suggest which skill a learner should practice next."
        )
    else:
        st.write(
            "The project goal was to reach around 95% accuracy, but the achieved result is lower than this target; the exact live values will appear here once the results JSON is present in the data folder."
        )

    st.subheader("Key challenges")
    st.markdown(
        "- Class imbalance in the learning interaction data, where correct answers can dominate the sequence history.\n"
        "- A small sequence-learning dataset with only 344 students and 15,061 interactions, which limits the model's ability to learn broad patterns.\n"
        "- Avoiding data leakage in engineered features by excluding hint and attempt count variables from the recommendation features.\n"
        "- Rare skills with very small totals producing nearly tied predictions because their global skill-difficulty estimates collapse to the same value when the sample is tiny."
    )

    st.subheader("Future scope")
    st.markdown(
        "- Classical ML baseline comparison page with Naive Bayes, Logistic Regression, Random Forest, and XGBoost.\n"
        "- Richer EDA once the raw pre-sampling dataset is added.\n"
        "- Multi-skill recommendation paths and sequence-aware routing across broader practice queues.\n"
        "- A/B testing recommendation strategies with real students and instructor-facing dashboards."
    )

    st.subheader("Applications")
    st.markdown(
        "- Automated skill gap detection for at-risk learners.\n"
        "- Personalized practice queues that adapt to each student's recent history.\n"
        "- Instructor-facing early-warning dashboards for intervention planning."
    )


def main():
    tabs = st.tabs(["Overview", "Dataset Explorer", "EDA Results", "Model Performance", "Try the Recommender", "Model Working", "Conclusion"])
    with tabs[0]:
        render_overview_tab()
    with tabs[1]:
        render_dataset_explorer_tab()
    with tabs[2]:
        render_eda_results_tab()
    with tabs[3]:
        render_model_performance_tab()
    with tabs[4]:
        render_recommender_tab()
    with tabs[5]:
        render_model_working_tab()
    with tabs[6]:
        render_conclusion_tab()


if __name__ == "__main__":
    main()
