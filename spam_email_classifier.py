# ============================================================
# 📧 SPAM EMAIL CLASSIFIER
# Uses Naive Bayes and SVM to classify messages as Spam or Ham
# ============================================================

# ---- Step 1: Install & Download Dataset --------------------
import os
import urllib.request

if not os.path.exists('spam.csv'):
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv',
        'spam.csv'
    )
    print("✅ Dataset downloaded!")
else:
    print("✅ Dataset already exists!")

# ---- Step 2: Import Libraries ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

print("✅ Libraries imported!")

# ---- Step 3: Load Dataset ----------------------------------
df = pd.read_csv('spam.csv', sep='\t', header=None, names=['label', 'message'])
print("\nDataset Shape:", df.shape)
print("\nLabel Counts:")
print(df['label'].value_counts())
print("\nSample Data:")
print(df.head())

# ---- Step 4: Visualize -------------------------------------
df['label'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'])
plt.title('Ham vs Spam Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('spam_vs_ham.png')
print("\n✅ Chart saved as spam_vs_ham.png")

# ---- Step 5: Preprocess ------------------------------------
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label_num'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
print("✅ Data preprocessed and ready!")

# ---- Step 6: Naive Bayes -----------------------------------
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
nb_preds = nb.predict(X_test_vec)

print("\n" + "="*40)
print("📊 NAIVE BAYES RESULTS")
print("="*40)
print(f"Accuracy: {round(accuracy_score(y_test, nb_preds)*100, 2)}%")
print(classification_report(y_test, nb_preds, target_names=['Ham', 'Spam']))

# ---- Step 7: SVM -------------------------------------------
svm = LinearSVC()
svm.fit(X_train_vec, y_train)
svm_preds = svm.predict(X_test_vec)

print("="*40)
print("📊 SVM RESULTS")
print("="*40)
print(f"Accuracy: {round(accuracy_score(y_test, svm_preds)*100, 2)}%")
print(classification_report(y_test, svm_preds, target_names=['Ham', 'Spam']))

# ---- Step 8: Test Your Own Messages ------------------------
def predict(msg):
    v = vectorizer.transform([msg])
    print(f"Message : {msg}")
    print(f"Naive Bayes → {'🚨 SPAM' if nb.predict(v)[0]==1 else '✅ HAM'}")
    print(f"SVM         → {'🚨 SPAM' if svm.predict(v)[0]==1 else '✅ HAM'}")
    print()

print("="*40)
print("🔍 SAMPLE PREDICTIONS")
print("="*40)
predict("Congratulations! You won a FREE iPhone. Click now!")
predict("Hey, are we meeting for lunch tomorrow?")
predict("URGENT! Your bank account needs verification now!")
predict("Get cheap medicines online! Limited offer!")
predict("Can you please send me the notes from today's class?")
