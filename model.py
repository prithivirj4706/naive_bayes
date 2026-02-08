import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load dataset
df = pd.read_csv("spam.csv")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df.Message, df.Category, test_size=0.2, random_state=5)

# Vectorization
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)

# Model training
model = MultinomialNB()
model.fit(X_train_count, y_train)

# Save vectorizer and model
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(v, f)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Saved vectorizer.pkl and model.pkl successfully")
