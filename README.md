# Sentiment Analysis Using NLP

## 1. Project Overview

This project focuses on classifying IMDb movie reviews as **Positive** or **Negative** using Natural Language Processing (NLP) and Machine Learning techniques.

I worked with the IMDb 50K Movie Reviews dataset, performed text preprocessing, converted reviews into numerical features using TF-IDF, and trained machine learning models for sentiment classification.

Two classification algorithms, Logistic Regression and Multinomial Naive Bayes, were implemented and evaluated to compare their performance.

## 2. Project Objectives

- Analyze and understand the IMDb movie review dataset.
- Perform exploratory data analysis (EDA).
- Clean and preprocess review text.
- Convert text into numerical features using TF-IDF.
- Train Logistic Regression and Multinomial Naive Bayes models.
- Evaluate and compare model performance.
- Analyze important words influencing sentiment predictions.
- Save the trained model and vectorizer for future predictions.

## 3. Dataset

**Dataset:** IMDb 50K Movie Reviews Dataset

**Source:** [Kaggle – IMDb 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

The dataset contains 50,000 movie reviews labelled as positive or negative.

**Dataset details after removing duplicates:**

| Description | Count |
|---|---:|
| Original reviews | 50,000 |
| Duplicate rows removed | 418 |
| Final reviews | 49,582 |
| Positive reviews | 24,884 |
| Negative reviews | 24,698 |

The dataset contains two columns: `review` (movie review text) and `sentiment` (positive or negative).

The dataset is not included in this repository. Download it from Kaggle and place `IMDB Dataset.csv` inside the `data/` directory.

## 4. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- NLTK
- Matplotlib and Seaborn
- Jupyter Notebook
- Joblib
- Git and GitHub

## 5. Project Structure

```text
Sentiment-Analysis-Using-NLP/
│
├── data/
│   └── IMDB Dataset.csv
│
├── models/
│   ├── best_sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_text_preprocessing.ipynb
│   └── 03_tfidf_vectorization.ipynb
│
├── reports/
│   └── figures/
│       ├── model_comparison.png
│       ├── roc_curve_comparison.png
│       ├── precision_recall_curve.png
│       └── feature_importance.png
│
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 6. Project Workflow

### Step 1: Exploratory Data Analysis

- Loaded and inspected the dataset.
- Checked missing values and duplicate records.
- Analyzed positive and negative review distribution.
- Examined review lengths and text patterns.
- Created visualizations to understand the dataset.

### Step 2: Text Preprocessing

The following preprocessing operations were performed:

- Removed HTML tags.
- Converted reviews to lowercase.
- Removed special characters.
- Cleaned extra spaces.

A new column named `cleaned_review` was created to store the processed text.

### Step 3: TF-IDF Vectorization

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert cleaned reviews into numerical feature vectors.

The vectorizer was configured with:

| Parameter | Value |
|---|---|
| max_features | 10,000 |
| ngram_range | (1, 2) |
| min_df | 2 |
| max_df | 0.95 |

Both unigrams and bigrams were considered during vectorization.

### Step 4: Train-Test Split

The dataset was divided into training and testing sets using stratified sampling.

- Training set: 80% (39,665 reviews)
- Testing set: 20% (9,917 reviews)

The TF-IDF vectorizer was fitted only on the training data and then used to transform the test data to avoid data leakage.

### Step 5: Model Training

Two classification algorithms were implemented:

**1. Logistic Regression**

Used as a classification model to predict whether a movie review expresses positive or negative sentiment.

**2. Multinomial Naive Bayes**

Implemented as a second model for comparison. It is commonly used for text classification tasks.

### Step 6: Model Evaluation

Both models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall AUC (PR-AUC)

### Step 7: Feature Analysis and Model Saving

- Analyzed Logistic Regression coefficients to identify words associated with positive and negative sentiment.
- Saved the trained Logistic Regression model and TF-IDF vectorizer using Joblib.
- Created `predict.py` to classify new movie reviews.

## 7. Model Performance Comparison

Both models were evaluated using 9,917 unseen movie reviews.

| Evaluation Metric | Logistic Regression | Multinomial Naive Bayes |
|---|---:|---:|
| Accuracy | 90.16% | 86.77% |
| Weighted F1-score | 90.16% | 86.77% |
| ROC-AUC | 96.38% | 94.13% |
| PR-AUC | 96.39% | 93.91% |

Based on the results, **Logistic Regression** was selected as the final model because it achieved better performance across all the evaluated metrics.

## 8. Model Evaluation Visualizations

### 8.1 Model Comparison

Compares Logistic Regression and Multinomial Naive Bayes across the evaluation metrics.

![Model Comparison](reports/figures/model_comparison.png)

### 8.2 ROC Curve

Shows the relationship between the True Positive Rate and False Positive Rate at different classification thresholds.

![ROC Curve](reports/figures/roc_curve_comparison.png)

### 8.3 Precision-Recall Curve

Shows the trade-off between precision and recall for both classification models.

![Precision-Recall Curve](reports/figures/precision_recall_curve.png)

### 8.4 Feature Importance

Displays words with strong positive and negative associations based on the Logistic Regression model coefficients.

![Feature Importance](reports/figures/feature_importance.png)

## 9. Running the Project Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/bharghavguru/Sentiment-Analysis-Using-NLP.git
cd Sentiment-Analysis-Using-NLP
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate the Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Prediction Script

```bash
python predict.py
```

The script accepts a movie review as input and predicts whether its sentiment is positive or negative. It also displays the model's estimated confidence score.

## 10. Example Prediction

**Input:**

```text
The movie was fantastic. The acting was brilliant and the story was amazing.
```

**Output:**

```text
Sentiment: POSITIVE
Model confidence: 95.68%
```

---

**Project Type:** Machine Learning | NLP | Text Classification | Sentiment Analysis

**Repository:** [Sentiment Analysis Using NLP](https://github.com/bharghavguru/Sentiment-Analysis-Using-NLP)