"""
NLP Preprocessing for Beginners using NLTK
This script demonstrates basic NLP preprocessing steps: tokenization, stopword removal, stemming, lemmatization, and POS tagging.
"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Sample text
sample_text = (
    "Natural Language Processing (NLP) is a fascinating field of artificial intelligence. "
    "It helps computers understand, interpret, and manipulate human language. "
    "The applications of NLP are growing rapidly, from chatbots to translation services."
)

print("\n--- Original Text ---\n")
print(sample_text)

# 1. Tokenization
print("\n--- Tokenization ---\n")
sentences = sent_tokenize(sample_text)
words = word_tokenize(sample_text)
print("Sentences:")
for i, sent in enumerate(sentences, 1):
    print(f"{i}. {sent}")
print("\nWords:", words)

# 2. Stopword Removal
print("\n--- Stopword Removal ---\n")
stop_words = set(stopwords.words('english'))
words_no_stop = [word for word in words if word.lower() not in stop_words and word.isalnum()]
print("Words after stopword removal:", words_no_stop)

# 3. Stemming
print("\n--- Stemming ---\n")
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in words_no_stop]
print("Stemmed words:", stemmed_words)

# 4. Lemmatization
print("\n--- Lemmatization ---\n")
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in words_no_stop]
print("Lemmatized words:", lemmatized_words)

'''
# 5. Part-of-Speech Tagging
print("\n--- Part-of-Speech (POS) Tagging ---\n")
pos_tags = nltk.pos_tag(words_no_stop)
for word, tag in pos_tags:
    print(f"{word:15} -> {tag}")

print("\n--- End of NLP Preprocessing Demo ---\n")
'''

# --- Load and preprocess legal_dataset.csv ---
print("\n--- Legal Dataset Preprocessing ---\n")
df = pd.read_csv('legal_dataset.csv')

# Choose a column to preprocess (e.g., first column)
text_col = df.columns[0]
df['tokenized'] = df[text_col].apply(lambda x: word_tokenize(str(x)))
stop_words = set(stopwords.words('english'))
df['no_stopwords'] = df['tokenized'].apply(lambda words: [w for w in words if w.lower() not in stop_words and w.isalnum()])
stemmer = PorterStemmer()
df['stemmed'] = df['no_stopwords'].apply(lambda words: [stemmer.stem(w) for w in words])
lemmatizer = WordNetLemmatizer()
df['lemmatized'] = df['no_stopwords'].apply(lambda words: [lemmatizer.lemmatize(w) for w in words])
#df['pos_tags'] = df['no_stopwords'].apply(lambda words: nltk.pos_tag(words))

# Save the processed DataFrame to a new CSV
df.to_csv('legal_dataset_preprocessed.csv', index=False)
print("Preprocessing complete. Saved to legal_dataset_preprocessed.csv.")
