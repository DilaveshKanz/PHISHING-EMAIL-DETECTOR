import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1) Load and swap so v1=text, v2=label
df = pd.read_csv('phishing.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['text', 'label']

# 2) Label is already numeric 0/1, so just cast to int
df['label'] = df['label'].astype(int)

# (Optional) Quick sanity check:
print("Unique labels:", df['label'].unique())  # should print [0] and [1]

# 3) Split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.2,
    random_state=42
)

# 4) Vectorize
vectorizer = TfidfVectorizer(max_features=4000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# 5) Train
model = LogisticRegression(C=10, max_iter=1500, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# 6) Evaluate
y_pred = model.predict(X_test_tfidf)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Precision:       {precision_score(y_test, y_pred):.2f}")
print(f"Recall:          {recall_score(y_test, y_pred):.2f}")
print(f"F1 Score:        {f1_score(y_test, y_pred):.2f}")

# 7) Save
joblib.dump({'model': model, 'vectorizer': vectorizer}, 'phishing_model.joblib')
