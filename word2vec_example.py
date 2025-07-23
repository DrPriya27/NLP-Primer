"""
Word2Vec Algorithm Overview
--------------------------
Word2Vec is a neural network-based algorithm for learning distributed representations of words (word embeddings).
It was introduced by Mikolov et al. (2013) and is widely used in NLP.

Algorithm Details:
There are two main architectures:
1. CBOW (Continuous Bag of Words):
   - Objective: Predict the target word from its context words.
   - For each position in the text, the model takes the context (surrounding words) and tries to predict the center word.
   - The context words are averaged (or summed) and fed into a shallow neural network to predict the target word.
   - Loss function: Negative log likelihood of the target word given the context.

2. Skip-gram:
   - Objective: Predict context words from the target word.
   - For each word in the text, the model tries to predict the surrounding context words.
   - The target word is fed into a shallow neural network to predict each context word.
   - Loss function: Negative log likelihood of the context words given the target word.

Both architectures use a shallow neural network with one hidden layer. The weights of the hidden layer become the word embeddings.
Training is typically done using stochastic gradient descent and techniques like negative sampling or hierarchical softmax to speed up computation.

CBOW is faster and works well with smaller datasets, while Skip-gram is better for rare words and larger datasets.

References:
- https://arxiv.org/pdf/1301.3781.pdf
- https://radimrehurek.com/gensim/models/word2vec.html

--------------------------
Real Example: Word2Vec Calculation
--------------------------
Suppose we have the following corpus:

    corpus = [
        'The quick brown fox jumps over the lazy dog',
        'Never jump over the lazy dog quickly',
        'A fox is quick and brown',
        'Dogs and foxes are animals',
        'Foxes are clever and quick',
        'Dogs are loyal and friendly'
    ]

We tokenize the sentences and train a Word2Vec model (CBOW):

    from gensim.models import Word2Vec
    import nltk
    nltk.download('punkt')
    tokenized_corpus = [nltk.word_tokenize(sentence.lower()) for sentence in corpus]
    model = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=3, min_count=1, sg=0, seed=42)

To get the vector for the word 'fox':

    vector = model.wv['fox']
    print(vector)

To find the most similar words to 'fox':

    print(model.wv.most_similar('fox'))

Sample Output:
    Vector for 'fox': [ 0.004 -0.012 ... 0.021 ]  # (50-dimensional vector)
    Most similar to 'fox': [('quick', 0.23), ('brown', 0.21), ...]

Word2Vec learns these vectors by training a shallow neural network to predict context words (Skip-gram) or the target word (CBOW) given its context. The resulting vectors capture semantic relationships, so similar words have similar vectors.

You can also perform analogies:

    result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'])
    print('king - man + woman =', result[0][0])

This finds the word whose vector is closest to king - man + woman (often 'queen').
"""

from gensim.models import Word2Vec, KeyedVectors
import nltk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Download punkt for tokenization
nltk.download('punkt')

# Sample corpus
corpus = [
    'The quick brown fox jumps over the lazy dog',
    'Never jump over the lazy dog quickly',
    'A fox is quick and brown',
    'Dogs and foxes are animals',
    'Foxes are clever and quick',
    'Dogs are loyal and friendly'
]

# Tokenize sentences
tokenized_corpus = [nltk.word_tokenize(sentence.lower()) for sentence in corpus]

# --- CBOW Example ---
# In gensim, CBOW is the default (sg=0)
cbow_model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=50,
    window=3,
    min_count=1,
    workers=1,
    sg=0,  # CBOW
    seed=42
)
cbow_model.save('word2vec_cbow.model')
cbow_loaded = Word2Vec.load('word2vec_cbow.model')
print("\nCBOW: Vector for 'fox':", cbow_loaded.wv['fox'])
print("CBOW: Most similar to 'fox':", cbow_loaded.wv.most_similar('fox'))

# --- Skip-gram Example ---
# In gensim, sg=1 means Skip-gram
skipgram_model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=50,
    window=3,
    min_count=1,
    workers=1,
    sg=1,  # Skip-gram
    seed=42
)
skipgram_model.save('word2vec_skipgram.model')
skipgram_loaded = Word2Vec.load('word2vec_skipgram.model')
print("\nSkip-gram: Vector for 'fox':", skipgram_loaded.wv['fox'])
print("Skip-gram: Most similar to 'fox':", skipgram_loaded.wv.most_similar('fox'))

# --- Visualization (CBOW) ---
words = list(cbow_loaded.wv.index_to_key)
word_vectors = cbow_loaded.wv[words]
pca = PCA(n_components=2)
result = pca.fit_transform(word_vectors)
plt.figure(figsize=(8,6))
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.title('Word2Vec Embeddings Visualization (CBOW, PCA)')
plt.show()

# --- Visualization (Skip-gram) ---
words_sg = list(skipgram_loaded.wv.index_to_key)
word_vectors_sg = skipgram_loaded.wv[words_sg]
pca_sg = PCA(n_components=2)
result_sg = pca_sg.fit_transform(word_vectors_sg)
plt.figure(figsize=(8,6))
plt.scatter(result_sg[:, 0], result_sg[:, 1])
for i, word in enumerate(words_sg):
    plt.annotate(word, xy=(result_sg[i, 0], result_sg[i, 1]))
plt.title('Word2Vec Embeddings Visualization (Skip-gram, PCA)')
plt.show()

# --- Analogy Example (CBOW) ---
if all(w in cbow_loaded.wv for w in ['king', 'man', 'woman']):
    result = cbow_loaded.wv.most_similar(positive=['king', 'woman'], negative=['man'])
    print('CBOW Analogy: king - man + woman =', result[0][0])

# --- Analogy Example (Skip-gram) ---
if all(w in skipgram_loaded.wv for w in ['king', 'man', 'woman']):
    result = skipgram_loaded.wv.most_similar(positive=['king', 'woman'], negative=['man'])
    print('Skip-gram Analogy: king - man + woman =', result[0][0])

# --- Using Pretrained Word2Vec (Google News, if available) ---
pretrained_path = 'GoogleNews-vectors-negative300.bin.gz'
if os.path.exists(pretrained_path):
    wv_pretrained = KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
    print('Pretrained vector for "dog":', wv_pretrained['dog'])
    print('Most similar to "dog":', wv_pretrained.most_similar('dog'))


from gensim.models import Word2Vec, KeyedVectors
import nltk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Download punkt for tokenization
nltk.download('punkt')

# Sample corpus
corpus = [
    'The quick brown fox jumps over the lazy dog',
    'Never jump over the lazy dog quickly',
    'A fox is quick and brown',
    'Dogs and foxes are animals',
    'Foxes are clever and quick',
    'Dogs are loyal and friendly'
]

# Tokenize sentences
tokenized_corpus = [nltk.word_tokenize(sentence.lower()) for sentence in corpus]

# Train Word2Vec model
w2v_model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=50,
    window=3,
    min_count=1,
    workers=1,
    seed=42
)

# Save and load model
w2v_model.save('word2vec_demo.model')
loaded_model = Word2Vec.load('word2vec_demo.model')

# Get vector for a word
print('Vector for "fox":', loaded_model.wv['fox'])

# Find most similar words
print('Most similar to "fox":', loaded_model.wv.most_similar('fox'))

# Visualize word embeddings using PCA
words = list(loaded_model.wv.index_to_key)
word_vectors = loaded_model.wv[words]
pca = PCA(n_components=2)
result = pca.fit_transform(word_vectors)
plt.figure(figsize=(8,6))
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.title('Word2Vec Embeddings Visualization (PCA)')
plt.show()

# Using Pretrained Word2Vec (Google News, if available)
# Download from https://code.google.com/archive/p/word2vec/ (if needed)
# Example: Load pretrained vectors (skip if not available)
pretrained_path = 'GoogleNews-vectors-negative300.bin.gz'
if os.path.exists(pretrained_path):
    wv_pretrained = KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
    print('Pretrained vector for "dog":', wv_pretrained['dog'])
    print('Most similar to "dog":', wv_pretrained.most_similar('dog'))

# Analogy: king - man + woman = ?
if 'king' in loaded_model.wv and 'man' in loaded_model.wv and 'woman' in loaded_model.wv:
    result = loaded_model.wv.most_similar(positive=['king', 'woman'], negative=['man'])
    print('king - man + woman =', result[0][0])

# More: Save vectors, use in downstream tasks, etc.