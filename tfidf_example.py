"""
TF-IDF Algorithm Overview
------------------------
TF-IDF (Term Frequency–Inverse Document Frequency) is a statistical measure used to evaluate how important a word is to a document in a collection or corpus.

Key Points:
- TF (Term Frequency): Measures how frequently a term occurs in a document.
- IDF (Inverse Document Frequency): Measures how important a term is, based on how common or rare it is across all documents.
- TF-IDF = TF * IDF
- Commonly used for text mining, information retrieval, and feature extraction in NLP.

References:
- https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

Practical Usage:
- Used to convert text documents into numerical feature vectors for ML models.
- Helps reduce the impact of common words and highlight important terms.

Real Example:
-------------
Suppose we have the following corpus:
    doc1 = "the cat sat on the mat"
    doc2 = "the dog sat on the log"
    doc3 = "cats and dogs are animals"

Let's calculate TF-IDF for the word "cat" in doc1:
- TF("cat", doc1) = 1/6 ("cat" appears once, 6 words in doc1)
- IDF("cat") = log(3 / (number of docs containing "cat")) = log(3/1) = 1.0986
- TF-IDF("cat", doc1) = TF * IDF = (1/6) * 1.0986 ≈ 0.183

This means "cat" is somewhat important in doc1, but if it appeared in all documents, its IDF would be lower, reducing its TF-IDF score.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample corpus
corpus = [
    'The quick brown fox jumps over the lazy dog',
    'Never jump over the lazy dog quickly',
    'A fox is quick and brown'
]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Convert to DataFrame for easy viewing
df_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print('TF-IDF Matrix:')
print(df_tfidf.round(2))

# Show vocabulary and IDF values
print('\nVocabulary:', vectorizer.vocabulary_)
print('IDF values:', dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_)))

# Example: Get TF-IDF score for 'fox' in each document
for i, doc in enumerate(corpus):
    print(f"TF-IDF for 'fox' in doc {i}: {df_tfidf.loc[i, 'fox']}")