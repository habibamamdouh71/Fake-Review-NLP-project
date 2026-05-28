# 🕵️ Fake Product Review Identification
### NLP Classification Project

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

This project aims to **automatically detect fake product reviews** using Natural Language Processing (NLP) and Machine Learning techniques. The dataset contains over **40,000 Amazon product reviews** labeled as either genuine (OR) or computer-generated/fake (CG).

---

## 🖥️ Application Preview

![Smart Text Classifier GUI](gui.png)

> Interactive web app built with **Streamlit** — classify any review as **Real (OR)** or **Fake (CG)** using 5 different ML models, with confidence scores and probability charts.

---

## 📂 Dataset

| Feature | Description |
|---|---|
| `category` | Product category (e.g., Electronics, Books, Home & Kitchen) |
| `rating` | Star rating (1–5) |
| `label` | `OR` = Original/Genuine · `CG` = Computer-Generated/Fake |
| `text` | Review text content |

- **Total records:** 40,432 → after cleaning: **40,420**
- **Classes:** Balanced — ~20,215 genuine vs ~20,205 fake
- **10 product categories:** Home & Kitchen, Electronics, Books, Sports & Outdoors, Movies & TV, Pet Supplies, Kindle Store, Tools & Home Improvement, Toys & Games, Clothing Shoes & Jewelry

---

## 🔧 Project Pipeline

```
Raw Data → Cleaning → EDA → Text Preprocessing → Feature Engineering → Modeling → Evaluation
```

### 1. 🧹 Data Cleaning
- Removed 12 duplicate rows
- Mapped labels: `CG → 0`, `OR → 1`
- Standardized category names and rating values
- Dropped rows with null values

### 2. 📊 Exploratory Data Analysis (EDA)
- Label distribution visualization
- Rating distribution per class
- Word frequency analysis
- Word clouds for fake vs genuine reviews

### 3. ✍️ Text Preprocessing
- Lowercasing
- Removing HTML tags, URLs, punctuation, and numbers
- Stopword removal (NLTK)
- Stemming (PorterStemmer / SnowballStemmer)
- Lemmatization (WordNetLemmatizer)

### 4. ⚙️ Feature Engineering
- **TF-IDF Vectorization** on review text
- Numerical features: `rating`, `review_length`, `word_count`
- Combined sparse matrix using `scipy.sparse.hstack`
- Label encoding with `LabelEncoder`

### 5. 🤖 Models Trained

| Model | Library |
|---|---|
| Logistic Regression | Scikit-learn |
| Naive Bayes (Multinomial) | Scikit-learn |
| Support Vector Machine (LinearSVC) | Scikit-learn |
| Random Forest | Scikit-learn |
| XGBoost | XGBoost |

### 6. 📈 Evaluation Metrics
- Accuracy Score
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix
- ROC-AUC Curve

---

## 🛠️ Technologies Used

```python
pandas · numpy · scikit-learn · xgboost
nltk · wordcloud · matplotlib · seaborn
scipy · joblib · streamlit · plotly
```

---

## 🌐 Web App Features (`app.py`)

| Feature | Details |
|---|---|
| 🧠 Classifier Page | Enter any review text and get instant prediction |
| 📊 Evaluation Page | Compare all 5 models — accuracy, precision, recall, F1 |
| 🤖 Model Selection | Choose between LR, NB, SVM, RF, XGBoost from sidebar |
| ⭐ Rating & Category | Include rating (1–5) and product category as features |
| 📈 Probability Chart | Visual confidence bar + Plotly probability chart |

---

## 📁 Project Structure

```
Fake-Review-NLP-project/
│
├── FAKE_REVIEW_PROJECT.ipynb     # Main notebook
├── app.py                        # Streamlit web app
├── fake reviews dataset.csv      # Dataset
│
├── model_logistic_regression.pkl
├── model_naive_bayes.pkl
├── model_svm.pkl
├── model_random_forest.pkl
├── model_xgboost.pkl
├── tfidf_vectorizer.pkl
├── label_encoder.pkl
├── gui.png                       # App screenshot
│
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/habibamamdouh71/Fake-Review-NLP-project.git
cd Fake-Review-NLP-project

# 2. Install required libraries
pip install pandas numpy scikit-learn xgboost nltk wordcloud matplotlib seaborn joblib scipy streamlit plotly

# 3. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# 4. Open the notebook
jupyter notebook FAKE_REVIEW_PROJECT.ipynb

# 5. Run the Streamlit app
streamlit run app.py
```

---

## Author

**Habiba Mamdouh**  
GitHub: [@habibamamdouh71](https://github.com/habibamamdouh71)

---

> ⭐ If you found this project helpful, please consider giving it a star!
