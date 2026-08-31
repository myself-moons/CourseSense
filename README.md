# CourseSense

CourseSense is a Streamlit dashboard for a personalized learning recommendation project built around a GRU-based knowledge-tracing model. It helps answer a practical question: given a student’s recent learning history, which skill should they practice next?

This project is fully grounded in the repository’s actual artifacts: engineered student data, sequence files, model weights, EDA outputs, and baseline comparison results. The app is designed to show the model, the data, the visual analysis, and the recommendation logic in one place.

## Project goal

The core idea is to estimate the probability that a student answers a math skill correctly, based on their recent interaction history, and then recommend the next best skill to practice.

The app supports two recommendation strategies:

- Remediation: target the student’s weakest skills
- ZPD: target a challenging but achievable skill, roughly around a 60% predicted success rate

## What the project includes

- A Streamlit dashboard in `app.py`
- GRU recommendation logic in `src/recommend.py`
- Training and modeling workflow in `src/gru_training.py`
- A reusable recommender prototype in `src/gru_recommender.py`
- Preprocessing and dataset cleaning logic in `src/preproessing.py`
- EDA plotting utilities in `src/eda_utils.py`
- Dataset/visualization helpers in `src/charts.py`
- Saved model and data artifacts under `models/` and `data/`

## Data preprocessing pipeline

The preprocessing step in `src/preproessing.py` is a critical part of the project. It does the following:

1. Downloads the raw ASSISTments dataset.
2. Keeps only the columns needed for sequence modeling.
3. Samples whole student sequences instead of random rows so the GRU keeps realistic student timelines.
4. Saves the raw sampled dataset as the BEFORE file.
5. Cleans the skill text by normalizing names, removing punctuation, dropping stop words, and lemmatizing the remaining words.
6. Saves the cleaned result as the AFTER file with a `clean_skill_name` column.

This preprocessing step makes the later exploratory analysis and recommendation modeling more stable and interpretable.

## How the model works

1. Student interaction data is loaded from the engineered feature table and sequence arrays.
2. The model tracks each student’s recent learning history over time.
3. A GRU processes that sequence and combines it with side features such as:
   - previous success/failure counts
   - rolling accuracy
   - skill difficulty
   - response time features
4. The model outputs a probability of correctness for the next interaction.
5. The recommender creates a “what-if” version for each candidate skill and asks the model, “If this student tries skill X next, how likely are they to succeed?”
6. The app ranks skills according to the selected strategy and displays the top recommendations.

## Repository structure

```text
CourseSense/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   ├── assistments_sample_AFTER.csv
│   ├── assistments_sample_BEFORE.csv
│   ├── engineered_features.csv
│   ├── gru_sequences.npz
│   ├── skill_vocab.json
│   ├── final_gru_results.json
│   └── baseline_results.json
├── models/
│   └── best_gru_model.keras
├── src/
│   ├── __init__.py
│   ├── charts.py
│   ├── eda.py
│   ├── eda_utils.py
│   ├── gru_recommender.py
│   ├── gru_training.py
│   ├── recommend.py
│   └── __pycache__/
└── .venv/
```

## Dashboard features

The app currently includes:

- Overview summary with project metrics
- Dataset explorer
- EDA Results visualizations
- Model Performance section with saved GRU metrics and baseline comparisons
- Recommendation engine with strategy selection
- Model Working explanation section
- Conclusion summary

## Local setup

```bash
cd /workspaces/CourseSense
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Run the app

```bash
cd /workspaces/CourseSense
source .venv/bin/activate
streamlit run app.py
```

## Notes

- The dashboard reads the project artifacts directly from `data/` and `models/`.
- If a metric file or model artifact is missing, the app warns gracefully instead of crashing.
- The recommendation logic filters out very rare skills so the model is not forced to rank poorly supported items.
- The project is data-backed and avoids inventing unsupported results or metrics.

## Project status

This project is complete as a working, data-backed demo and dashboard for personalized learning recommendations using a GRU-based model.
