import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("data/fake_or_real_news.csv")  # adjust the path

# Split data
X = df['text']
y = df['label'].map({'FAKE': 0, 'REAL': 1})  # Encoding labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Save model and vectorizer
with open("model/fake_news_model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)










from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model
with open("model/fake_news_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

@app.route('/')
def home():
    return '''
        <form action="/predict" method="post">
            <textarea name="news" rows="10" cols="60"></textarea><br>
            <input type="submit">
        </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    news_vec = vectorizer.transform([news])
    prediction = model.predict(news_vec)
    result = "REAL" if prediction[0] == 1 else "FAKE"
    return f"<h2>Prediction: {result}</h2>"

if __name__ == '__main__':
    app.run(debug=True)





import pickle

# Load model
with open("model/fake_news_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# Input your news article
news = """
A recent study claims that the moon landing in 1969 was staged by the government.
"""

# Transform and predict
news_vec = vectorizer.transform([news])
prediction = model.predict(news_vec)

print("Prediction:", "REAL" if prediction[0] == 1 else "FAKE")
