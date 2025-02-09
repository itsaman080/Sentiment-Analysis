### **📌 Twitter Sentiment Analysis**  

A machine learning-based **Twitter Sentiment Analysis** project that classifies tweets as **positive, negative, or neutral** using **Natural Language Processing (NLP) and Machine Learning algorithms**.  

---

## **🚀 Features**  
✅ **Classifies tweets** into **positive, negative, or neutral** sentiments  
✅ **Preprocesses and cleans tweets** with tokenization, stopword removal, and stemming  
✅ **Extracts insights** from tweet data using **word clouds, frequency distributions, and visualizations**  
✅ **Trains multiple ML models** including **Random Forest, Logistic Regression, Decision Tree, SVM, and XGBoost**  
✅ **Evaluates performance** using **accuracy, F1-score, and confusion matrix**  

---

## **📂 Dataset**  
- **Train Dataset:** `train_tweet.csv` (Labeled tweets with sentiment labels)  
- **Test Dataset:** `test_tweets.csv` (Unlabeled tweets for prediction)  

---

## **🛠 Technologies Used**  
- **Programming Language:** Python 🐍  
- **Libraries:**  
  - **Data Processing:** Pandas, NumPy  
  - **NLP:** NLTK, Gensim, WordCloud  
  - **Feature Extraction:** Scikit-learn (CountVectorizer, TfidfVectorizer)  
  - **Machine Learning:** Scikit-learn, XGBoost  
  - **Visualization:** Matplotlib, Seaborn  

---

## **📌 Project Workflow**  

1️⃣ **Data Loading:** Import and explore the dataset  
2️⃣ **Data Cleaning & Preprocessing:** Remove noise, tokenize, remove stopwords, apply stemming  
3️⃣ **Feature Engineering:** Create **Bag of Words (BoW)** and **Word2Vec embeddings**  
4️⃣ **Exploratory Data Analysis (EDA):** Generate **word clouds, hashtag analysis, tweet length distribution**  
5️⃣ **Model Training:** Train models (**Random Forest, Logistic Regression, SVM, Decision Tree, XGBoost**)  
6️⃣ **Evaluation:** Use **F1-score, accuracy, and confusion matrix** for model performance analysis  
7️⃣ **Prediction on Test Data**  

---

## **📊 Model Performance**  

| Model                  | Training Accuracy | Validation Accuracy | F1-Score |
|------------------------|------------------|----------------------|----------|
| Random Forest         | **High**         | **Good**             | ✅ **High** |
| Logistic Regression   | **Moderate**     | **Good**             | ✅ **Moderate** |
| Decision Tree         | **High**         | **Overfitting Risk** | ⚠️ **Varies** |
| SVM                   | **High**         | **Good**             | ✅ **High** |
| XGBoost               | **Best**         | **Best**             | ⭐ **Best** |

---

## **📜 Results & Insights**  

📌 **Most Frequent Words:**  
- Word cloud visualization highlights commonly used words in **positive and negative tweets**  

📌 **Most Used Hashtags:**  
- Top hashtags extracted from neutral and negative tweets help analyze trends  

📌 **Model Performance:**  
- **XGBoost and SVM** delivered the best accuracy and F1-score  

---

## **📌 Future Enhancements**  
🔹 Improve sentiment detection with **LSTM / Transformers (BERT, RoBERTa)**  
🔹 Implement **real-time Twitter streaming API** for live sentiment analysis  
🔹 Deploy as a **web application** using Flask or Streamlit  

---

## **📬 Contact**   
🔗 **LinkedIn:** [Connect](https://linkedin.com/in/aman-kumar-thakur)  

---
