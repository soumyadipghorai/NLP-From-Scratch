import os
import gensim
from gensim.models import LsiModel
from gensim import models
from gensim import corpora
from gensim.utils import lemmatize
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords, stem_text
from gensim.parsing.preprocessing import strip_numeric
import pandas as pd
from gensim import similaritiesc

model_dir = '/home/ramaseshan/temp/'
physics_corpus_dir = 'https://raw.githubusercontent.com/Ramaseshanr/anlp/master/corpus/phy_corpus.txt'


cor = pd.read_csv(physics_corpus_dir, sep='\n', header=None)[0]


def preprocessing():
    for document in cor:
        doc = strip_numeric(stem_text(document))
        yield gensim.utils.tokenize(doc, lower=True)

texts = preprocessing()
dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=1, keep_n=700)

doc_term_matrix = [dictionary.doc2bow(tokens) for tokens in preprocessing()]
tfidf = models.TfidfModel(doc_term_matrix)
corpus_tfidf = tfidf[doc_term_matrix]

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=1)  # initialize an LSI transformation
doc = 'car acceleration speed far'
vec_bow = dictionary.doc2bow(doc.lower().split())

vec_lsi = lsi[vec_bow]  # convert the query to LSI space
index = similarities.MatrixSimilarity(lsi[doc_term_matrix])
unsorted_similarity = index[vec_lsi]
sorted_similarity = sorted(enumerate(unsorted_similarity), key=lambda item: -item[1])
for index, similarity in sorted_similarity:
    print(similarity, cor[index])