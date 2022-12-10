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
Other references:
* https://medium.com/web-mining-is688-spring-2021/cosine-similarity-and-tfidf-c2a7079e13fa 
* https://iq.opengenus.org/document-similarity-tf-idf/ 
* https://github.com/parthasm/Search-Engine-TF-IDF

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
Other references:
* https://jalammar.github.io/illustrated-word2vec/
* https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial
* https://medium.com/@makcedward/how-negative-sampling-work-on-word2vec-7bf8d545b116
* https://www.linkedin.com/posts/jalammar_the-illustrated-word2vec-a-gentle-intro-activity-6977227264614666240-t9WD/?utm_source=share&utm_medium=member_android

### 4. Global Vectors for word representation [(GloVe)](https://nlp.stanford.edu/pubs/glove.pdf)
It is also based on creating contextual word embeddings. Word2Vec is a window-based method, in which the model relies on local information for generating word embeddings, which in turn is limited to the window size that we choose. GloVe on the other hand captures both global and local statistics in order to come up with the word embeddings.


https://neptune.ai/blog/vectorization-techniques-in-nlp-guide

### 6. Doc2Vec

### References
* https://www.linkedin.com/posts/puspanjalisarma_nlp-datascience-artificialintelligence-activity-6981164409419149313-fwH9/?utm_source=share&utm_medium=member_android
* https://medium.com/nlplanet/two-minutes-nlp-11-word-embeddings-models-you-should-know-a0581763b9a9
* https://www.linkedin.com/posts/activity-6974001281262604288-2g3Y/?utm_source=share&utm_medium=member_android
* [Word2Vec Vs BERT] https://www.linkedin.com/posts/lavanyats_bert-vs-word2vec-embeddings-activity-6978221983201185793-aiat/?utm_source=share&utm_medium=member_android

[NLP] SpaCy Classifier with pre-train token2vec VS. One without pre-train | by Tom Lin | Towards Data Science

𝗳𝗲𝘄-𝘀𝗵𝗼𝘁 𝗰𝗹𝗮𝘀𝘀𝗶𝗳𝗶𝗰𝗮𝘁𝗶𝗼𝗻 𝗼𝗳 𝘁𝗮𝗯𝘂𝗹𝗮𝗿 𝗱𝗮𝘁𝗮 𝘄𝗶𝘁𝗵 𝗟𝗮𝗿𝗴𝗲 𝗟𝗮𝗻𝗴𝘂𝗮𝗴𝗲 𝗠𝗼𝗱𝗲𝗹𝘀
https://www.linkedin.com/posts/abdalimran_artificialintelligence-naturallanguageprocessing-activity-6989123329202827264-8D91/?utm_source=share&utm_medium=member_android

sara and another conference
https://www.linkedin.com/posts/synopsys_iccad2022-semiconductor-activity-6991513416850956288-iOit/?utm_source=share&utm_medium=member_android



https://verificationacademy.com/forums/uvm/multiple-register-design

https://www.google.com/search?q=ral+automation+eda&sxsrf=ALiCzsa8G-Brwis-SzRFB3P-GNDmuuXYXA%3A1667570710939&ei=FhxlY4_-OIGE4-EPi8qYwAI&oq=ral+&gs_lcp=ChNtb2JpbGUtZ3dzLXdpei1zZXJwEAEYADIECCMQJzIECCMQJzIECCMQJzIFCAAQkQIyCggAELEDEIMBEEMyBQgAEJECMgQIABBDMgoIABCxAxCDARBDOgoIABBHENYEELADOgcIIxDqAhAnOgsIABCABBCxAxCDAToICAAQsQMQgwE6CwguEIAEELEDEIMBOgUIABCABDoHCAAQsQMQQzoFCC4QgAQ6CwguEIMBELEDEIAEOg0ILhDHARDRAxDUAhBDOgcILhDUAhBDSgQIQRgAULERWPwjYNoraAVwAHgAgAH5AYgBmAqSAQUwLjEuNZgBAKABAbABD8gBCMABAQ&sclient=mobile-gws-wiz-serp

https://www.libhunt.com/l/python/topic/uvm-ral-model
https://learnuvmverification.com/index.php/2020/11/02/how-to-implement-uvm-ral-part-1/
https://scholar.google.co.in/scholar?start=0&q=Register+Automation+using+Machine+Learning+at+DVCon+2019.&hl=en&as_sdt=0,5&as_vis=1
https://www.agnisys.com/register-automation-using-machine-learning/




ARM AMBA 5
https://www.linkedin.com/posts/venkatesh-kudumula-46550014_synopsys-introduces-the-industrys-first-activity-6993154923798413312-jJaK/?utm_source=share&utm_medium=member_android

PUZZLEX AND good profile for advisory point of view
https://www.linkedin.com/posts/synopsys_semiconductor-ai-chiplet-activity-6992908180070518784-jD3Y/?utm_source=share&utm_medium=member_android


AI/ML IN DVCON
https://blogs.sw.siemens.com/verificationhorizons/2020/01/21/ai-ml-at-dvcon-from-theory-to-applications/

AI CAN DIagnose diaase based on voice
https://www.linkedin.com/posts/synopsys_artificial-intelligence-could-soon-diagnose-activity-6986740763531382784-OOae/?utm_source=share&utm_medium=member_android

Electronic Design Process Symposium (EDPS)
https://www.linkedin.com/posts/kerimgenc_impostersyndrome-innovation-medical-activity-6983826394443042816-mJdX/?utm_source=share&utm_medium=member_android

Table to text
https://www.linkedin.com/posts/abdalimran_artificialintelligence-datascience-machinelearning-activity-6989601483277160449-Muks/?utm_source=share&utm_medium=member_android


synopsys people
https://www.linkedin.com/posts/andrewbolster_ml-datascience-workanniversary-activity-6988101997300219904-DPJI/?utm_source=share&utm_medium=member_android

verification checks identification in electroninc design automation with nlp - Google Search
rule based heuristic for verification aspects identification in electronic design automati nlpn - Google Search
verification requirement extraction from specification - Google Search
a fully automated approach to requirements extraction from design document - Google Search
automatic requirement extraction in electronic design automation - Google Search

Machine Learning Electronic Design Automation:Unlocking Designs | Advanced PCB Design Blog | Cadence

What Is an AI Accelerator? – How It Works | Synopsys


automatic translation of design specification - Google Search

AI-Based Sequence Detection for IP and SoC Verification and (edacafe.com)

Synopsys talks AI in verification at DVCon – Tech Design Forum (techdesignforums.com)

Machine Learning Gives Us EDA Tools That Can Learn - Digital Engineering 24/7 (digitalengineering247.com)

Enhancing Chip Verification with AI and Machine Learning (synopsys.com)


electronic-design-automation · GitHub Topics

formal verification protocol checks + machine leaeming - Google Search

spacyspaCy 101: Everything you need to know · spaCy Usage Documentation

Deepak ahuja
Green changed sentences Red is for tablesBlue condition need to be taken care in code

Parth to add vocabulary in bert model
Softlink To do and ignore there are two features in verdi 

What is Intelligent Document Processing(IDP)? - Benefits and use-cases of IDP in different industries (docsumo.com)

Ucie make df2 to df3.

amazon go
https://www.youtube.com/watch?v=NrmMk1Myrxcamazon go
nlp - Text Classification with Spacy : going beyond the basics to improve performance - Stack Overflow

Text Classification and ML Model Interpretation with ELi5,Sklearn and SpaCy – JCharisTech

Classification of Tweets using SpaCy - Analytics Vidhya

machine learning - Improving text classification & labeling in imbalanced dataset - Data Science Stack Exchange

How To Build an Effective Email Spam Classification model with Spacy Python (dataaspirant.com)

Extract and classify text of common documents | Data Science and Machine Learning | Kaggle

Kaggle's NLP: Text Classification – Weights & Biases (wandb.ai)

machine learning - How to train a spacy model for text classification? - Data Science Stack Exchange

Text Classification: All Tips and Tricks from 5 Kaggle Competitions | Neptune Blog

Text Classification using SpaCy | Kaggle


Dfi - _n is for active low means when zero hai to value jayegi. _N ni h to active high signal h.  Highlighting in pdf
Ddr and dfi mai wherever we have a parameter written one req min.
Issue of table header not spanninh multiple pages  two ai


https://www.kaggle.com/code/andreshg/nlp-glove-bert-tf-idf-lstm-explained

https://github.com/cmhungsteve/LeetCode

Easy Guide To Data Preprocessing In Python - KDnuggets

Data Preprocessing for Machine Learning (codesource.io)


OFFICE:
Sentence Transformers: Sentence-BERT - Sentence Embeddings using Siamese BERT-Networks arXiv #demo - YouTube

NLP with Disaster Tweets - EDA, Cleaning and BERT | Kaggle

AND ONE IN PHONE SCREENSHOT
Nlp disaster tweet ke sari notebook padoSearch similarity kaggle per dekhoOverleaf ke paper read karoDatacamp videos
Deployment kaise har spec per. Table template kaise sabki. Mipi per table excel file.

office:Knowledge seriessemiconductor talk series youtubebenefits vides youtube

Gray are not checks. Blue are checks but line items are not sufficient. Green bold are checks. Yellow ones even not clear to him

https://blogs-synopsys-com.cdn.ampproject.org/c/s/blogs.synopsys.com/from-silicon-to-software/2022/05/26/synopsys-learning-center/amp/

https://www.src.org/calendar/e006556/

DVCON
https://www.google.com/search?q=machine+learning+%2B+verification+electronic+design+automation&client=ms-android-samsung-ga-rev1&sxsrf=ALiCzsaaMdbKeeLjIkBidSwdGOoi2qspHg%3A1661881769874&ei=qU0OY9X5NL6NseMP1-uQ4Ag&oq=&gs_lcp=ChNtb2JpbGUtZ3dzLXdpei1zZXJwEAEYATIHCCMQ6gIQJzIHCCMQ6gIQJzIHCCMQ6gIQJzIHCCMQ6gIQJzIHCCMQ6gIQJzIHCCMQ6gIQJzIHCCMQ6gIQJzIHCCMQ6gIQJzIHCCMQ6gIQJzIHCCMQ6gIQJzIHCCMQ6gIQJzIHCCMQ6gIQJzIHCCMQ6gIQJzIHCCMQ6gIQJzIHCCMQ6gIQJzoHCAAQRxCwAzoHCAAQsAMQQzoSCC4QxwEQ0QMQyAMQsAMQQxgBOg8ILhDUAhDIAxCwAxBDGAFKBQg4EgExSgQIQRgAUABYAGCBGmgDcAF4AIABAIgBAJIBAJgBAKABAbABD8gBD8ABAdoBBAgBGAg&sclient=mobile-gws-wiz-serp#ip=1

https://firsteda.com/news/agnisys-blog-post-register-automation-using-machine-learning/

https://openreview.net/forum?id=oIhzg4GJeOf

https://www.linkedin.com/posts/christineayoung_how-to-accelerate-the-soc-design-flow-with-activity-6970792950779838464-7rYt/?utm_source=share&utm_medium=member_android

https://www.linkedin.com/posts/stelix_an-eda-ai-master-class-by-synopsys-ceo-aart-activity-6966462357090779137-dFx-/?utm_source=linkedin_share&utm_medium=android_app

https://www.linkedin.com/posts/synopsys_ai-aihardwaresummit-activity-6965004262875553792-aGen/?utm_source=linkedin_share&utm_medium=android_app

https://sjobs.brassring.com/TGNewUI/Search/Home/HomeWithPreLoad?partnerid=25235&siteid=5359&jobid=1958321&PageType=jobdetails#jobDetails=1958321_5359


Data extraction from a PDF table with semi-structured layout | by Volodymyr Holomb | Towards Data Science

How To Clean Up Large PDF Datasets (investintech.com)

deep learning extract text from pdf - Google Search

NLP analysis of PDF documents | Kaggle

wldmrgml/main - Jovian


3 ways to scrape tables from PDFs with Python - Open Source Automation (theautomatic.net)

Python Camelot borderless table extraction issue - Stack Overflow

What are the best libraries for table extraction from a pdf document? (researchgate.net)

Advanced Usage — Camelot 0.10.1 documentation (camelot-py.readthedocs.io)

python - Camelot switches characters around - Stack Overflow

Python Camelot borderless table extraction issue - Stack Overflow

camelot/camelot at master · atlanhq/camelot · GitHub

https://www.techdesignforums.com/practice/technique/executable-specification-soc-ip/

https://www.google.com/search?q=idesignspec&client=ms-android-samsung-ga-rev1&sxsrf=ALiCzsbCaI8dGrP_CMeiBDoBUOVw-XkvFw%3A1667928888139&ei=OJNqY4iRCN_az7sPtISi2AI&oq=idesugnspev&gs_lcp=ChNtb2JpbGUtZ3dzLXdpei1zZXJwEAEYADIHCAAQgAQQDTIGCAAQHhANMgYIABAeEA0yBQgAEKIEMgUIABCiBDIFCAAQogQyBQgAEKIEOgoIABBHENYEELADOgcIIxDqAhAnOgQIIxAnOgUIABCRAjoLCAAQgAQQsQMQgwE6EQguEIAEELEDEIMBEMcBENEDOgQIABBDOggIABCxAxCRAjoHCAAQsQMQQzoHCC4QsQMQQzoFCAAQgAQ6DAgAELEDEAIQnwEQQzoHCAAQgAQQCjoFCC4QgAQ6DQgAEIAEELEDEIMBEAo6CggAEIAEELEDEA1KBAhBGABQihRY8SZg-TpoA3AAeACAAfEBiAGPEZIBBTAuOC4zmAEAoAEBsAEPyAEFwAEB&sclient=mobile-gws-wiz-serp#ip=1






