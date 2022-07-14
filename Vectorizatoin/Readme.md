# What is vectorization? 
Vectorization is a process of converting input text data into vectors of real numbers which is the format that ML models support. 

## Vectorization techniques

### 1. Bag of Words
It involves three major steps [Tokenization](https://neptune.ai/blog/tokenization-in-nlp), Vocabulary creation and Vector creation (via considering frequency of vocabulary words in a given document).

```
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer() # for unigram
# use cv = CountVectorizer(ngram_range=(2,2)) for bigram and above
X = cv.fit_transform(input_data) 
X = X.toarray()

#print vocabulary
sorted(cv.vocabulary_.keys())
```

### 2. Term Frequency–Inverse Document Frequency (TF-IDF)

TF stands for Term Frequency. It can be understood as a normalized frequency score. IDF is a reciprocal of the Document Frequency. Please refer to [article](https://neptune.ai/blog/vectorization-techniques-in-nlp-guide) for more details.


```
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(input_data)

#Creating a data frame with feature names, i.e. the words, as indices, and sorted TF-IDF scores as a column:
df = pandas.DataFrame(X[0].T.todense(), index=tfidf.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)

```
Since the transformed TFIDF feature matrix comes out as a Scipy Compressed Sparse Row matrix, which can’t be viewed in its raw form, we have converted it into a Numpy array, via todense() operation after taking its transform. Similarly, we get the complete vocabulary of tokenized words via get_feature_names().

### 3. [Word2Vec](https://arxiv.org/pdf/1301.3781.pdf)
Neural Network based method to generate [word embeddings](https://neptune.ai/blog/word-embeddings-guide). In earlier two methods, semantics were completely ignored. With the introduction of Word2Vec, the vector representation of words was said to be contextually aware, probably for the first time ever.

```
from gensim import models

# Google’s pre-trained model 
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
gzip -d GoogleNews-vectors-negative300.bin.gz
w2v = models.KeyedVectors.load_word2vec_format(
'./GoogleNews-vectors-negative300.bin', binary=True)
vect = w2v['healthy']
w2v.most_similar('happy')
sents = ['coronavirus is a highly infectious disease',
   'coronavirus affects older people the most', 
   'older people are at high risk due to this disease']
# training dataset in form of a list of lists of tokenized sentences
sents = [sent.split() for sent in sents]
custom_model = models.Word2Vec(sents, min_count=1,size=300,workers=4)

```
good visualization [link](https://ronxin.github.io/wevi/)

### 4. Global Vectors for word representation [(GloVe)](https://nlp.stanford.edu/pubs/glove.pdf)
It is also based on creating contextual word embeddings. Word2Vec is a window-based method, in which the model relies on local information for generating word embeddings, which in turn is limited to the window size that we choose. GloVe on the other hand captures both global and local statistics in order to come up with the word embeddings.


https://neptune.ai/blog/vectorization-techniques-in-nlp-guide

