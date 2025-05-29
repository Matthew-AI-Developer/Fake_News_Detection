import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df_real = pd.read_csv("G:/python/ML/Fake_News_Detection/True.csv")
df_fake = pd.read_csv("G:/python/ML/Fake_News_Detection/Fake.csv")

df_real["label"] = 1
df_fake["label"] = 0

df = pd.concat([df_real, df_fake], axis=0).reset_index(drop=True)

df.drop(columns=["subject", "date"], inplace=True)

plt.figure(figsize=(6, 4))
df["label"].value_counts().plot(kind="bar", color=["red", "green"])
plt.xticks(ticks=[0, 1], labels=["Fake News", "Real News"], rotation=0)
plt.xlabel("News Type")
plt.ylabel("Count")
plt.title("Distribution of Fake and Real News")
plt.show()

X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
