# CourseSense 🎓
### Sentiment-Aware Course Review Classification with Classical ML + GRU

CourseSense analyzes real Coursera course reviews to classify learner sentiment (Negative / Neutral / Positive), compares classical machine learning models against a GRU-based deep learning model, and surfaces the results in an interactive dashboard — with an eye toward powering personalized course recommendations based on predicted learner sentiment.

---

## 📌 Project Overview

Online learning platforms generate huge volumes of free-text feedback that's hard to act on manually. This project builds an end-to-end NLP pipeline that:

1. Cleans and prepares raw student course reviews
2. Engineers text features using classical (BoW/TF-IDF) and embedding-based (Word2Vec) methods
3. Trains and compares multiple classifiers, with a **GRU (Gated Recurrent Unit) neural network** as the primary sequential model
4. Evaluates all models on standard classification metrics
5. Presents everything — data, models, results — in an interactive Power BI dashboard

The end goal: a model that can flag negative learner sentiment early and support course recommendation/improvement decisions.

---

## 📂 Dataset

- **Source:** [Coursera Course Reviews](https://github.com/sharmaroshan/Coursera-Reviews-Analysis) (`reviews.csv`) — scraped course reviews with 1–5 star ratings
- **Original size:** 107,018 reviews
- **Working sample:** 9,000 reviews, stratified and class-balanced (3,000 each of Negative / Neutral / Positive), sampled from reviews with 3+ words, to keep training fast and metrics meaningful across all three classes rather than skewed toward the dominant 5-star class
- **Labels:** Star rating (1–5) mapped to 3-class sentiment:
  - 1–2★ → **Negative**
  - 3★ → **Neutral**
  - 4–5★ → **Positive**

---

## 🗺️ Project Roadmap

- [x] **Dataset sourcing & sampling** — pulled full dataset, explored class distribution, built a balanced 9,000-review sample
- [x] **Text preprocessing** — lowercasing, URL/punctuation removal, tokenization, stopword removal, lemmatization
- [ ] **Exploratory Data Analysis** — rating/sentiment distribution, review length patterns, word frequency by class
- [ ] **Feature engineering** — CountVectorizer / TF-IDF for classical models, Word2Vec / trainable embeddings for the GRU
- [ ] **Classical model development** — at least 3 of: Naïve Bayes, Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost
- [ ] **GRU model development** — Embedding → GRU → Dense classifier over sentiment classes
- [ ] **Model evaluation & comparison** — Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC curve
- [ ] **Analytical dashboard (Power BI)** — dataset overview, key stats, interactive charts, word cloud, sentiment distribution, model comparison, conclusion

---

## 🧰 Tech Stack

- **Language:** Python 3
- **NLP:** NLTK (tokenization, stopwords, lemmatization), scikit-learn (CountVectorizer, TF-IDF), Gensim (Word2Vec)
- **Modeling:** scikit-learn (Naïve Bayes, Logistic Regression, Random Forest, SVM), XGBoost, TensorFlow/Keras (GRU)
- **Dashboard:** Power BI

---

## 📁 Repository Structure

```
coursesense/
├── data/
│   ├── coursera_reviews_sample_BEFORE.csv   # raw sampled reviews (9,000 rows)
│   └── coursera_reviews_sample.csv          # cleaned/preprocessed reviews
├── preprocessing/
│   └── text_preprocessing.py                # text cleaning pipeline
├── notebooks/                               # EDA, feature engineering, model training (in progress)
├── dashboard/                               # Power BI file (in progress)
└── README.md
```

---

## 📊 Planned Conclusion Highlights

- **Key findings:** patterns in what drives negative vs. positive course feedback, model performance comparison, whether GRU meaningfully outperforms classical baselines on this data
- **Challenges faced:** natural class imbalance in the raw data (91% positive reviews), short/noisy review text, tuning a sequence model on a modest dataset size
- **Future scope:** extending to multi-class star-rating prediction, incorporating course metadata, testing transformer-based models (BERT) as a stretch goal
- **Applications:** automated flagging of at-risk courses, sentiment-informed course recommendations, instructor feedback triage at scale

---

## ⏱️ Status

Work in progress — dataset and preprocessing complete, modeling and dashboard stages underway.
