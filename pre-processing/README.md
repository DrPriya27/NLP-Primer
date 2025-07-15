# NLP Tutorial for Beginners: Using NLTK and spaCy

This tutorial introduces basic Natural Language Processing (NLP) tasks using two popular Python libraries: NLTK and spaCy. It covers installation, tokenization, stopword removal, stemming, lemmatization, part-of-speech tagging, and named entity recognition.

---


## 1. Installation

Before you start, you need to install the required libraries. NLTK (Natural Language Toolkit) and spaCy are both available via pip. Additionally, spaCy requires you to download a language model (here, the small English model).

```bash
pip install nltk spacy
python -m spacy download en_core_web_sm
```
This will install both libraries and the English language model for spaCy.

---


## 2. Importing Libraries

After installation, import the necessary modules from both libraries. These modules provide functions for tokenization, stopword removal, stemming, and lemmatization.

```python
import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
```

---

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

## 3. Downloading NLTK Data

NLTK relies on several datasets for its core functions. You need to download these datasets the first time you use NLTK. 'punkt' is used for tokenization, 'stopwords' for filtering common words, 'wordnet' for lemmatization, and 'averaged_perceptron_tagger' for part-of-speech tagging.

```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

---


## 4. Tokenization

Tokenization is the process of splitting text into smaller units, such as words or sentences. This is a fundamental step in NLP, as most tasks require working with individual words or sentences.

### NLTK
```python
text = "Natural Language Processing (NLP) is fun! Let's learn it."
words = word_tokenize(text)  # Splits text into words
sentences = sent_tokenize(text)  # Splits text into sentences
print("Words:", words)
print("Sentences:", sentences)
```
`word_tokenize` uses pre-trained models to split text into words, handling punctuation and contractions. `sent_tokenize` splits text into sentences.

### spaCy
```python
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
print("Words:", [token.text for token in doc])
print("Sentences:", [sent.text for sent in doc.sents])
```
In spaCy, you process text with the `nlp` object, which returns a `Doc` object. You can iterate over tokens (words) and sentences directly.

---


## 5. Stopword Removal

Stopwords are common words (like "the", "is", "and") that often do not add significant meaning to text analysis. Removing them helps focus on the important words in your data.

### NLTK
```python
stop_words = set(stopwords.words('english'))  # Get the list of English stopwords
filtered_words = [w for w in words if w.lower() not in stop_words]  # Remove stopwords
print("Filtered Words:", filtered_words)
```
This code filters out stopwords from the tokenized words.

### spaCy
```python
filtered_words_spacy = [token.text for token in doc if not token.is_stop]  # Remove stopwords using spaCy's built-in attribute
print("Filtered Words (spaCy):", filtered_words_spacy)
```
spaCy marks stopwords with the `is_stop` attribute, making it easy to filter them out.

---


## 6. Stemming and Lemmatization

Both stemming and lemmatization are techniques to reduce words to their root form. Stemming simply chops off word endings, while lemmatization uses vocabulary and morphological analysis to return the base or dictionary form of a word.

### Stemming (NLTK)
```python
ps = PorterStemmer()
stemmed = [ps.stem(w) for w in filtered_words]
print("Stemmed:", stemmed)
```
Stemming can be useful for quick-and-dirty text normalization, but it may produce non-words.

### Lemmatization (NLTK)
```python
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w) for w in filtered_words]
print("Lemmatized:", lemmatized)
```
Lemmatization returns valid words and is generally preferred for tasks where word meaning matters.

### Lemmatization (spaCy)
```python
lemmas_spacy = [token.lemma_ for token in doc]
print("Lemmas (spaCy):", lemmas_spacy)
```
spaCy provides lemmatization out of the box via the `lemma_` attribute of each token.

---


## 7. Part-of-Speech (POS) Tagging

Part-of-speech (POS) tagging assigns a grammatical category (noun, verb, adjective, etc.) to each word. This is useful for understanding sentence structure and for downstream tasks like parsing and information extraction.

### NLTK
```python
pos_tags = nltk.pos_tag(words)
print("POS Tags:", pos_tags)
```
`pos_tag` returns a list of tuples, each containing a word and its POS tag.

### spaCy
```python
for token in doc:
    print(token.text, token.pos_, token.tag_)
```
spaCy provides both a simple POS tag (`pos_`) and a more detailed tag (`tag_`).

---


## 8. Named Entity Recognition (NER)

Named Entity Recognition (NER) is the process of identifying and classifying named entities in text, such as people, organizations, locations, dates, etc. This is a key step in extracting structured information from unstructured text.

### spaCy
```python
for ent in doc.ents:
    print(ent.text, ent.label_)
```
Each entity has a text span and a label (such as PERSON, ORG, GPE, DATE, etc.).

---


## 9. Summary Table

| Task                | NLTK Functionality         | spaCy Functionality         |
|---------------------|---------------------------|----------------------------|
| Tokenization        | word_tokenize, sent_tokenize | nlp(text), doc.sents      |
| Stopword Removal    | stopwords.words           | token.is_stop              |
| Stemming            | PorterStemmer             | Not available              |
| Lemmatization       | WordNetLemmatizer         | token.lemma_               |
| POS Tagging         | pos_tag                   | token.pos_, token.tag_      |
| NER                 | Not built-in              | doc.ents                   |

---


## 10. References
- [NLTK Documentation](https://www.nltk.org/)
- [spaCy Documentation](https://spacy.io/)
