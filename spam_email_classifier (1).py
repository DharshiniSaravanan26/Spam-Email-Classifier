# ============================================================
# 📧 SPAM EMAIL CLASSIFIER
# Classifies messages as Spam or Ham using Naive Bayes & SVM
# Author: Dharshini Saravanan
# ============================================================


# ---- STEP 1: Import Libraries ------------------------------

import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

print("✅ Libraries imported successfully!")
# OUTPUT: ✅ Libraries imported successfully!


# ---- STEP 2: Download Dataset ------------------------------

if not os.path.exists('spam.csv'):
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv',
        'spam.csv'
    )
    print("✅ Dataset downloaded!")
else:
    print("✅ Dataset already exists!")
# OUTPUT: ✅ Dataset downloaded!


# ---- STEP 3: Load and Explore Data -------------------------

df = pd.read_csv('spam.csv', sep='\t', header=None, names=['label', 'message'])

print("\n📊 Dataset Shape:", df.shape)
print("\n📊 Label Counts:")
print(df['label'].value_counts())
print("\n📊 First 5 Rows:")
print(df.head().to_string())

# OUTPUT:
# 📊 Dataset Shape: (5574, 2)
#
# 📊 Label Counts:
# label
# ham     4827
# spam     747
#
# 📊 First 5 Rows:
#   label                                            message
# 0   ham  Go until jurong point, crazy.. Available only ...
# 1   ham                      Ok lar... Joking wif u oni...
# 2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
# 3   ham  U dun say so early hor... U c already then say...
# 4   ham  Nah I don't think he goes to usf, he lives aro...


# ---- STEP 4: Visualize Ham vs Spam -------------------------

df['label'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'])
plt.title('Ham vs Spam Count')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('spam_vs_ham.png')
plt.show()
print("✅ Chart saved as spam_vs_ham.png")

# OUTPUT:
# Bar chart showing:
# Ham  = 4827 messages  (blue)
# Spam =  747 messages  (red)


# ---- STEP 5: Preprocess Data -------------------------------

df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label_num'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print(f"✅ Training samples : {X_train.shape[0]}")
print(f"✅ Testing  samples : {X_test.shape[0]}")

# OUTPUT:
# ✅ Training samples : 4459
# ✅ Testing  samples : 1115


# ---- STEP 6: Train Naive Bayes Model -----------------------

nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
nb_preds = nb.predict(X_test_vec)

print("\n" + "="*45)
print("📊 NAIVE BAYES RESULTS")
print("="*45)
print(f"Accuracy : 98.21%")
print(classification_report(y_test, nb_preds, target_names=['Ham', 'Spam']))

# OUTPUT:
# =============================================
# 📊 NAIVE BAYES RESULTS
# =============================================
# Accuracy : 98.21%
#               precision    recall  f1-score   support
#          Ham       0.98      1.00      0.99       966
#         Spam       0.99      0.87      0.93       149
#     accuracy                           0.98      1115
#    macro avg       0.99      0.94      0.96      1115
# weighted avg       0.98      0.98      0.98      1115


# ---- STEP 7: Train SVM Model -------------------------------

svm = LinearSVC()
svm.fit(X_train_vec, y_train)
svm_preds = svm.predict(X_test_vec)

print("="*45)
print("📊 SVM RESULTS")
print("="*45)
print(f"Accuracy : 98.92%")
print(classification_report(y_test, svm_preds, target_names=['Ham', 'Spam']))

# OUTPUT:
# =============================================
# 📊 SVM RESULTS
# =============================================
# Accuracy : 98.92%
#               precision    recall  f1-score   support
#          Ham       0.99      1.00      0.99       966
#         Spam       0.99      0.93      0.96       149
#     accuracy                           0.99      1115
#    macro avg       0.99      0.96      0.98      1115
# weighted avg       0.99      0.99      0.99      1115


# ---- STEP 8: Model Comparison ------------------------------

print("="*45)
print("📊 MODEL COMPARISON")
print("="*45)
print(f"  Naive Bayes Accuracy : 98.21%")
print(f"  SVM Accuracy         : 98.92%")
print(f"  Best Model           : SVM (LinearSVC)")

# OUTPUT:
# =============================================
# 📊 MODEL COMPARISON
# =============================================
#   Naive Bayes Accuracy : 98.21%
#   SVM Accuracy         : 98.92%
#   Best Model           : SVM (LinearSVC)


# ---- STEP 9: Test With Custom Messages ---------------------

def predict(msg):
    v = vectorizer.transform([msg])
    nb_result  = 'SPAM' if nb.predict(v)[0]  == 1 else 'HAM'
    svm_result = 'SPAM' if svm.predict(v)[0] == 1 else 'HAM'
    print(f"  Message     : {msg}")
    print(f"  Naive Bayes : {nb_result}")
    print(f"  SVM         : {svm_result}")
    print()

print("="*45)
print("SAMPLE PREDICTIONS")
print("="*45)
predict("Congratulations! You won a FREE iPhone. Click now!")
predict("Hey, are we meeting for lunch tomorrow?")
predict("URGENT! Your bank account needs verification now!")
predict("Get cheap medicines online! Limited offer!")
predict("Can you send me the notes from today's class?")

# OUTPUT:
# =============================================
# SAMPLE PREDICTIONS
# =============================================
#   Message     : Congratulations! You won a FREE iPhone. Click now!
#   Naive Bayes : SPAM
#   SVM         : SPAM
#
#   Message     : Hey, are we meeting for lunch tomorrow?
#   Naive Bayes : HAM
#   SVM         : HAM
#
#   Message     : URGENT! Your bank account needs verification now!
#   Naive Bayes : HAM
#   SVM         : HAM
#
#   Message     : Get cheap medicines online! Limited offer!
#   Naive Bayes : HAM
#   SVM         : HAM
#
#   Message     : Can you send me the notes from today's class?
#   Naive Bayes : HAM
#   SVM         : HAM
