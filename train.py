import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings("ignore")

from utils import load_data, preprocess_text

def run_pipeline():
    print("========================================")
    print("PHASE 1: DATA PROCESSING & EDA")
    print("========================================")
    data_path = 'data/jira_dataset.csv'
    df = load_data(data_path)
    print(f"Data shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    df.drop_duplicates(inplace=True)
    
    # Priority distribution
    os.makedirs('notebooks', exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='priority', order=['critical', 'high', 'medium', 'low'])
    plt.title("Priority Distribution")
    dist_path = 'notebooks/priority_distribution.png'
    plt.savefig(dist_path)
    plt.close()
    print(f"Saved EDA plot to {dist_path}")
    
    # Task length analysis
    df['task_length'] = df['clean_summary'].apply(lambda x: len(str(x).split()))
    print(f"Average task length (words): {df['task_length'].mean():.2f}")
    
    print("\n========================================")
    print("PHASE 2: NLP PREPROCESSING")
    print("========================================")
    print("Sample before:")
    print(df['clean_summary'].iloc[:2].tolist())
    df['clean_text'] = df['clean_summary'].apply(preprocess_text)
    print("Sample after (lowercase, no punctuations, tokenized, stopword removed, lemmatized):")
    print(df['clean_text'].iloc[:2].tolist())
    
    clean_data_path = 'data/cleaned_jira_dataset.csv'
    df.to_csv(clean_data_path, index=False)
    print(f"Cleaned dataset saved to {clean_data_path}")
    
    print("\n========================================")
    print("PHASE 3 & 4: FEATURE ENGINEERING AND MODEL BUILDING")
    print("========================================")
    X = df['clean_text']
    y = df['priority']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    print("TF-IDF Vectorization complete (max_features=5000).")
    
    models = {
        'Naive Bayes': MultinomialNB(),
        'SVM (LinearSVC)': LinearSVC(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    best_model = None
    best_acc = 0
    best_model_name = ""
    
    print("\n========================================")
    print("PHASE 5: MODEL EVALUATION")
    print("========================================")
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        print(f"--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_model_name = name
            
    print(f"\n=> Best Model selected: {best_model_name} with Accuracy: {best_acc:.4f} <=")
    
    # Detailed report for best model
    y_pred_best = best_model.predict(X_test_vec)
    print("\nClassification Report (Best Model):")
    print(classification_report(y_test, y_pred_best))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_best, labels=['critical', 'high', 'medium', 'low'])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['critical', 'high', 'medium', 'low'], yticklabels=['critical', 'high', 'medium', 'low'], cmap='Blues')
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_path = 'notebooks/confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved Confusion Matrix to {cm_path}")
    
    print("\n========================================")
    print("PHASE 6: MODEL SAVING")
    print("========================================")
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
    print("Model and vectorizer saved successfully into 'models/' folder.")

if __name__ == '__main__':
    run_pipeline()
