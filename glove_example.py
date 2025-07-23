"""
GloVe Algorithm Overview
-----------------------
GloVe (Global Vectors for Word Representation) is an unsupervised learning algorithm for obtaining vector representations for words.
It was developed by Stanford (Pennington et al., 2014).

Algorithm Details:
- GloVe builds a global word-word co-occurrence matrix from the corpus.
- For each word pair (i, j), it counts how often word j appears in the context of word i.
- The model learns word vectors such that the dot product of two word vectors approximates the logarithm of the probability of their co-occurrence:
    w_i^T w_j + b_i + b_j ≈ log(X_ij)
  where X_ij is the co-occurrence count, w_i and w_j are word vectors, and b_i, b_j are biases.
- The objective function minimizes the weighted least squares error over all word pairs.
- GloVe is a count-based model (unlike predictive models like Word2Vec).

References:
- https://nlp.stanford.edu/projects/glove/
- https://nlp.stanford.edu/pubs/glove.pdf

Practical Usage:
- Pretrained GloVe vectors are available for many languages and domains.
- Common use: Load pretrained vectors and use them for NLP tasks (similarity, analogy, downstream ML, etc).

-----------------------
Real Example: GloVe Calculation
-----------------------
Suppose we have downloaded pretrained GloVe vectors (e.g., glove.6B.50d.txt).

    # Load GloVe vectors
    glove_vectors = {}
    with open('glove.6B.50d.txt', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_vectors[word] = vector

To get the vector for the word 'fox':

    vec_fox = glove_vectors['fox']
    print(vec_fox)

To compute cosine similarity between 'fox' and 'dog':

    from numpy.linalg import norm
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    print(cosine_similarity(glove_vectors['fox'], glove_vectors['dog']))

To find the most similar words to 'dog':

    def most_similar(word, vectors, topn=5):
        word_vec = vectors[word]
        similarities = {}
        for w, v in vectors.items():
            if w != word:
                similarities[w] = cosine_similarity(word_vec, v)
        return sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:topn]
    print(most_similar('dog', glove_vectors))

To perform analogies (king - man + woman = ?):

    def analogy(word_a, word_b, word_c, vectors):
        result_vec = vectors[word_a] - vectors[word_b] + vectors[word_c]
        similarities = {}
        for w, v in vectors.items():
            similarities[w] = cosine_similarity(result_vec, v)
        # Exclude input words
        for w in [word_a, word_b, word_c]:
            similarities.pop(w, None)
        return sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:1]
    print(analogy('king', 'man', 'woman', glove_vectors))

Sample Output:
    Cosine similarity between fox and dog: 0.78
    Most similar to "dog": [('dogs', 0.92), ('puppy', 0.89), ...]
    king - man + woman = queen

GloVe vectors can be used for similarity, analogy, visualization, and as features in downstream ML tasks.
"""

import numpy as np
import requests, zipfile, io, os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Download GloVe embeddings (small sample for demo)
glove_file = 'glove.6B.50d.txt'
if not os.path.exists(glove_file):
    url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extract(glove_file)

# Load GloVe vectors
glove_vectors = {}
with open(glove_file, encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        glove_vectors[word] = vector

# Example: Get vector for 'fox' and find cosine similarity with 'dog'
from numpy.linalg import norm
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

vec_fox = glove_vectors['fox']
vec_dog = glove_vectors['dog']
print('Cosine similarity between fox and dog:', cosine_similarity(vec_fox, vec_dog))

# Find most similar words to 'dog' using GloVe
def most_similar(word, vectors, topn=5):
    word_vec = vectors[word]
    similarities = {}
    for w, v in vectors.items():
        if w != word:
            similarities[w] = cosine_similarity(word_vec, v)
    return sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:topn]

print('Most similar to "dog":', most_similar('dog', glove_vectors))

# Analogy: king - man + woman = ?
def analogy(word_a, word_b, word_c, vectors):
    result_vec = vectors[word_a] - vectors[word_b] + vectors[word_c]
    similarities = {}
    for w, v in vectors.items():
        similarities[w] = cosine_similarity(result_vec, v)
    # Exclude input words
    for w in [word_a, word_b, word_c]:
        similarities.pop(w, None)
    return sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:1]

if all(w in glove_vectors for w in ['king', 'man', 'woman']):
    print('king - man + woman =', analogy('king', 'man', 'woman', glove_vectors)[0][0])

# Visualize GloVe embeddings using PCA
words = ['dog', 'fox', 'king', 'queen', 'man', 'woman', 'animal', 'friendly', 'loyal', 'quick']
word_vecs = np.array([glove_vectors[w] for w in words if w in glove_vectors])
pca = PCA(n_components=2)
result = pca.fit_transform(word_vecs)
plt.figure(figsize=(8,6))
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate([w for w in words if w in glove_vectors]):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.title('GloVe Embeddings Visualization (PCA)')
plt.show()

# Integration: Use GloVe vectors in downstream ML tasks (e.g., averaging for document vectors)
def document_vector(doc, vectors):
    words = doc.lower().split()
    valid_vecs = [vectors[w] for w in words if w in vectors]
    if valid_vecs:
        return np.mean(valid_vecs, axis=0)
    else:
        return np.zeros(vectors[next(iter(vectors))].shape)

sample_doc = 'The quick brown fox jumps over the lazy dog'
doc_vec = document_vector(sample_doc, glove_vectors)
print('Document vector shape:', doc_vec.shape)
