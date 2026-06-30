# 📝 Text Classification with Naive Bayes

## 🎯 Project Overview

This project implements a text classification system using **Multinomial Naive Bayes** algorithm to categorize text documents into different topics. The model achieves **88.24% accuracy** on a synthetic text dataset covering multiple categories.

## 🏆 Key Achievements

| Metric | Value |
|--------|-------|
| **Accuracy** | **88.24%** |
| Model Type | Multinomial Naive Bayes |
| Feature Extraction | CountVectorizer (Bag of Words) |
| Dataset Size | 85 samples |

## 📊 Dataset

The dataset contains **85** text samples classified into **4 categories**:

| Category | Description | Examples |
|----------|-------------|----------|
| Technology | AI, computing, innovation | "Artificial intelligence is advancing..." |
| Sports | Football, tournaments, athletes | "Football fans are excited about..." |
| Politics | Policies, climate change, governance | "New policies regarding climate change..." |
| Entertainment | Movies, music, pop culture | "The latest blockbuster movie..." |

### Dataset Statistics
- **Total samples**: 85
- **Training samples**: 68 (80%)
- **Testing samples**: 17 (20%)
- **Categories**: 4
- **No missing values**

## 🧠 Methodology

### 1. Data Preprocessing

```python
# Text vectorization using Bag of Words
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
