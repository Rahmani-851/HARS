import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import joblib

print("Starting the model training process")

# Data Loading and Preprocessing
try:
    fake_df = pd.read_csv('Fake.csv')
    true_df = pd.read_csv('True.csv')
except FileNotFoundError:
    print("Error: Make sure 'Fake.csv' and 'True.csv' are in the same directory.")
    exit()

fake_df['label'] = 1
true_df['label'] = 0

df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df['content'] = df['title'] + ' ' + df['text']

X = df['content']
y = df['label']

print("Data loaded and preprocessed.")

# Data Splitting and Feature Extraction
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

print("TF-IDF Vectorizer has been fitted.")

# Model Training
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_tfidf, y_train)
    # Save each trained model
    joblib.dump(model, f'{name.lower().replace(" ", "_")}.joblib')
    print(f"{name} trained and saved.")

# Save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
print("TF-IDF Vectorizer saved.")

print("\nAll models and the vectorizer have been successfully trained and saved.")