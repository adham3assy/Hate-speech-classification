# 🛡️ Hate Speech Classification on Twitter

This repository provides a complete pipeline for classifying hate speech in tweets using both traditional machine learning models and deep learning approaches. It includes NLP-specific preprocessing, custom transformers, pre-trained word embeddings, and techniques to address data imbalance.

---

## 📁 Dataset

- Tweets labeled as **hate speech** or **non-hate speech**.

---

## 📚 Project Structure

- `Traditional_Models.ipynb`  
  Implements classical ML models (e.g., Logistic Regression, SVM, Random Forest) using various vectorization techniques.

- `Deep_Learning_Models.ipynb`  
  Uses RNN-based architectures:
  - Simple RNN
  - LSTM
  - GRU
  - ConvLSTM
  - Bidirectional versions

---

## ⚙️ Techniques & Tools

### 🧹 NLP Preprocessing
- Lowercasing, removing mentions, hashtags, links
- Tokenization and lemmatization
- Stopword removal
- Regex-based cleaning

### 🧠 Feature Engineering
- Custom vectorization transformers
- Embedding techniques:
  - Bag of Words
  - TF-IDF
  - Pretrained embeddings:
    - Word2Vec (via Gensim)
    - GloVe
    - FastText

### 🧪 Model Evaluation
- Macro-averaged **F1-score**
- Precision, recall, accuracy
- Cross-validation

### ⚖️ Imbalanced Data Handling
- **SMOTE** oversampling
- **Class weighting** in DL models

---

## 🧰 Libraries Used

- `scikit-learn`
- `tensorflow` / `keras`
- `gensim`, `spacy`, `nltk`
- `pandas`, `numpy`, `matplotlib`
- `regex`

---

