"""
Doc2Vec Algorithm Overview
-------------------------
Doc2Vec (Paragraph Vector) is an extension of Word2Vec for learning vector representations of documents (sentences, paragraphs, or entire texts).
It was introduced by Le & Mikolov (2014).

Algorithm Details:
Doc2Vec learns fixed-length feature representations for variable-length pieces of text. There are two main models:
1. Distributed Memory (DM):
   - Objective: Predict a target word using both context words and a unique document id.
   - The document vector acts as a memory that provides additional context for prediction (similar to CBOW).
   - Both word vectors and document vectors are updated during training.
   - Loss function: Negative log likelihood of the target word given context and document id.

2. Distributed Bag of Words (DBOW):
   - Objective: Predict words in the document using only the document id (ignores context).
   - The model tries to predict words randomly sampled from the document using the document vector (similar to Skip-gram).
   - Only document vectors are updated during training.
   - Loss function: Negative log likelihood of the context words given the document id.

Doc2Vec is useful for document similarity, classification, clustering, and retrieval.

References:
- https://arxiv.org/pdf/1405.4053.pdf
- https://radimrehurek.com/gensim/models/doc2vec.html

-------------------------
Real Example: Doc2Vec Calculation
-------------------------
Suppose we have the following corpus:

    corpus = [
        'The quick brown fox jumps over the lazy dog',
        'Never jump over the lazy dog quickly',
        'A fox is quick and brown',
        'Dogs and foxes are animals',
        'Foxes are clever and quick',
        'Dogs are loyal and friendly'
    ]

We tag each document with a unique id and train a Doc2Vec model (DM):

    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    import nltk
    nltk.download('punkt')
    tagged_data = [TaggedDocument(words=nltk.word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(corpus)]
    model = Doc2Vec(vector_size=50, window=2, min_count=1, workers=1, epochs=40, seed=42, dm=1)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

To get the vector for the first document:

    vector = model.dv['0']
    print(vector)

To find the most similar documents to the first document:

    print(model.dv.most_similar('0'))

To infer a vector for a new document:

    new_doc = 'A quick animal jumps over a lazy dog'
    new_vec = model.infer_vector(nltk.word_tokenize(new_doc.lower()))
    print(new_vec)
    print(model.dv.most_similar([new_vec]))

Sample Output:
    Vector for first document: [ 0.01 -0.02 ... 0.03 ]  # (50-dimensional vector)
    Most similar to first document: [('2', 0.88), ('4', 0.85), ...]
    Inferred vector for new document: [ 0.02 -0.01 ... 0.04 ]
    Most similar to new document: [('0', 0.90), ('3', 0.87), ...]

Doc2Vec learns these vectors by training a shallow neural network to predict words using document context. The resulting vectors capture semantic relationships between documents.
"""

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

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

# Tag documents
# Each document is tagged with a unique id
# TaggedDocument(words, [tag])
tagged_data = [TaggedDocument(words=nltk.word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(corpus)]

# --- Distributed Memory (DM) Example ---
doc2vec_dm = Doc2Vec(vector_size=50, window=2, min_count=1, workers=1, epochs=40, seed=42, dm=1)
doc2vec_dm.build_vocab(tagged_data)
doc2vec_dm.train(tagged_data, total_examples=doc2vec_dm.corpus_count, epochs=doc2vec_dm.epochs)
print("\nDM: Vector for first document:", doc2vec_dm.dv['0'])
print("DM: Most similar to first document:", doc2vec_dm.dv.most_similar('0'))

# --- Distributed Bag of Words (DBOW) Example ---
doc2vec_dbow = Doc2Vec(vector_size=50, window=2, min_count=1, workers=1, epochs=40, seed=42, dm=0)
doc2vec_dbow.build_vocab(tagged_data)
doc2vec_dbow.train(tagged_data, total_examples=doc2vec_dbow.corpus_count, epochs=doc2vec_dbow.epochs)
print("\nDBOW: Vector for first document:", doc2vec_dbow.dv['0'])
print("DBOW: Most similar to first document:", doc2vec_dbow.dv.most_similar('0'))

# --- Infer vector for a new document ---
new_doc = 'A quick animal jumps over a lazy dog'
new_vec_dm = doc2vec_dm.infer_vector(nltk.word_tokenize(new_doc.lower()))
print('DM: Inferred vector for new document:', new_vec_dm)
print('DM: Most similar to new document:', doc2vec_dm.dv.most_similar([new_vec_dm]))

new_vec_dbow = doc2vec_dbow.infer_vector(nltk.word_tokenize(new_doc.lower()))
print('DBOW: Inferred vector for new document:', new_vec_dbow)
print('DBOW: Most similar to new document:', doc2vec_dbow.dv.most_similar([new_vec_dbow]))

# --- Visualization (DM) ---
doc_tags = [str(i) for i in range(len(corpus))]
doc_vectors_dm = [doc2vec_dm.dv[tag] for tag in doc_tags]
pca_dm = PCA(n_components=2)
result_dm = pca_dm.fit_transform(doc_vectors_dm)
plt.figure(figsize=(8,6))
plt.scatter(result_dm[:, 0], result_dm[:, 1])
for i, tag in enumerate(doc_tags):
    plt.annotate(f'Doc {tag}', xy=(result_dm[i, 0], result_dm[i, 1]))
plt.title('Doc2Vec DM Document Embeddings Visualization (PCA)')
plt.show()

# --- Visualization (DBOW) ---
doc_vectors_dbow = [doc2vec_dbow.dv[tag] for tag in doc_tags]
pca_dbow = PCA(n_components=2)
result_dbow = pca_dbow.fit_transform(doc_vectors_dbow)
plt.figure(figsize=(8,6))
plt.scatter(result_dbow[:, 0], result_dbow[:, 1])
for i, tag in enumerate(doc_tags):
    plt.annotate(f'Doc {tag}', xy=(result_dbow[i, 0], result_dbow[i, 1]))
plt.title('Doc2Vec DBOW Document Embeddings Visualization (PCA)')
plt.show()

# --- Using Pretrained Doc2Vec (if available) ---
pretrained_path = 'pretrained_doc2vec.model'
if os.path.exists(pretrained_path):
    pretrained_doc2vec = Doc2Vec.load(pretrained_path)
    print('Pretrained vector for a sample document:', pretrained_doc2vec.infer_vector(nltk.word_tokenize('sample text')))


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

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

# Tag documents
tagged_data = [TaggedDocument(words=nltk.word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(corpus)]

# Train Doc2Vec model
doc2vec_model = Doc2Vec(vector_size=50, window=2, min_count=1, workers=1, epochs=40, seed=42)
doc2vec_model.build_vocab(tagged_data)
doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

# Save and load model
doc2vec_model.save('doc2vec_demo.model')
loaded_doc2vec = Doc2Vec.load('doc2vec_demo.model')

# Get vector for first document
print('Vector for first document:', loaded_doc2vec.dv['0'])

# Find most similar documents
print('Most similar to first document:', loaded_doc2vec.dv.most_similar('0'))

# Infer vector for a new document
new_doc = 'A quick animal jumps over a lazy dog'
new_vec = loaded_doc2vec.infer_vector(nltk.word_tokenize(new_doc.lower()))
print('Inferred vector for new document:', new_vec)

# Find most similar documents to the new document
print('Most similar to new document:', loaded_doc2vec.dv.most_similar([new_vec]))

# Visualize document embeddings using PCA
doc_tags = [str(i) for i in range(len(corpus))]
doc_vectors = [loaded_doc2vec.dv[tag] for tag in doc_tags]
pca = PCA(n_components=2)
result = pca.fit_transform(doc_vectors)
plt.figure(figsize=(8,6))
plt.scatter(result[:, 0], result[:, 1])
for i, tag in enumerate(doc_tags):
    plt.annotate(f'Doc {tag}', xy=(result[i, 0], result[i, 1]))
plt.title('Doc2Vec Document Embeddings Visualization (PCA)')
plt.show()

# Using Pretrained Doc2Vec (if available)
# Example: Load pretrained vectors (skip if not available)
pretrained_path = 'pretrained_doc2vec.model'
if os.path.exists(pretrained_path):
    pretrained_doc2vec = Doc2Vec.load(pretrained_path)
    print('Pretrained vector for a sample document:', pretrained_doc2vec.infer_vector(nltk.word_tokenize('sample text')))

# More: Save vectors, use in downstream tasks, etc.
