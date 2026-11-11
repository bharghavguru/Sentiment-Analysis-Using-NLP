# Sentiment Analysis Using NLP

A Machine Learning project that classifies IMDb movie reviews as **Positive** or **Negative** using Natural Language Processing (NLP).

## Project Overview

This project uses the IMDb 50K Movie Reviews dataset to build a sentiment classification model.

It applies text preprocessing, TF-IDF vectorization, and machine learning algorithms to identify the sentiment expressed in movie reviews.

The final model is Logistic Regression, selected after comparing its performance with Multinomial Naive Bayes.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- NLTK
- Matplotlib and Seaborn
- Jupyter Notebook
- Joblib

## Dataset

**IMDb 50K Movie Reviews Dataset**

[Kaggle Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

- Original reviews: 50,000
- Duplicate rows removed: 418
- Final reviews: 49,582
- Positive reviews: 24,884
- Negative reviews: 24,698

The dataset is not included in this repository. Download it from Kaggle and place `IMDB Dataset.csv` inside the `data/` folder.

## Project Workflow

1. Exploratory Data Analysis (EDA)
2. Text preprocessing
   - HTML tag removal
   - Lowercase conversion
   - Special character removal
   - Whitespace normalization
3. TF-IDF vectorization
4. Train-test splitting (80:20)
5. Model training
   - Logistic Regression
   - Multinomial Naive Bayes
6. Model evaluation
7. Feature importance analysis
8. Saving the trained model
9. Predicting sentiment from new reviews

## Model Performance

The models were evaluated on 9,917 unseen test reviews.

| Metric | Logistic Regression | Multinomial Naive Bayes |
|---|---:|---:|
| Accuracy | 90.16% | 86.77% |
| Weighted F1-score | 90.16% | 86.77% |
| ROC-AUC | 96.38% | 94.13% |
| PR-AUC | 96.39% | 93.91% |

**Final Model: Logistic Regression**

Logistic Regression achieved better performance across the reported evaluation metrics and was selected as the final model.

## Visualizations

### Model Comparison
![Model Comparison](reports/figures/model_comparison.png)

### ROC Curve
![ROC Curve](reports/figures/roc_curve_comparison.png)

### Precision-Recall Curve
![Precision Recall Curve](reports/figures/precision_recall_curve.png)

### Feature Importance
![Feature Importance](reports/figures/feature_importance.png)

## Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/bharghavguru/Sentiment-Analysis-Using-NLP.git
cd Sentiment-Analysis-Using-NLP
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run sentiment prediction

```bash
python predict.py
```

Enter a movie review when prompted. The program returns its predicted sentiment and model confidence.

## Example Prediction

**Input:**

```text
The movie was fantastic. The acting was brilliant and the story was amazing.
```

**Output:**

```text
Sentiment: POSITIVE
Model confidence: 95.68%
```

## Repository Structure

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

## Limitations

- The model classifies reviews into only two categories: Positive and Negative.
- Neutral sentiment is not supported.
- The model uses TF-IDF and traditional machine learning rather than transformer-based language models.
- Prediction confidence represents the model's estimated probability, not a guarantee of correctness.

## Future Improvements

- Experiment with advanced NLP models such as LSTM and BERT.
- Improve text preprocessing and feature extraction.
- Explore hyperparameter tuning.
- Investigate handling of neutral and mixed sentiment.

## Author

**GitHub:** [bharghavguru](https://github.com/bharghavguru)