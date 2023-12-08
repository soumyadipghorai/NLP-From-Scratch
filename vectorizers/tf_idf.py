import nltk 
import numpy as np
import pandas as pd 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
nltk.download('stopwords')

stop_words_list = list(stopwords.words('english')) 

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

"""
* in the column you have the words, in the row index is the docs. each value = tf * idf 
* tf = count of that term in that doc/ length of the doc  
* df = count of docs containing the term 
* idf = log(N/df) + 1 ==> N total docs 

* here you can pass either the text content in a corpus 
* or you can pass in file names 
* or you can pass in file objects with a read method. 
* text can be transformed to lowercase and can pass custom stop_words list. 
* also set binary = True for only binary matrix representation
* ngram_range will give the (min_range, max_range) ==> (1, 2) means unigram and bigrams
* max_features = n will select top n features based on term frequency accross the corpus  
* use idf = False will set idf = 1 and give the tf matrix
* smooth_idf will make idf = log(N/(df+1)) + 1
* norm will add different normalization on it. None for no normalization
"""

print('\n')
print("#"*20, " CORPUS ", "#"*20)

for text in corpus :
    print(text)

print("#"*20, " CORPUS ", "#"*20)
print('\n')

#? tfidf vector without stop words 
count_vector = CountVectorizer(
    input = 'content', lowercase= True, stop_words= [], 
    encoding= ' utf-8', ngram_range=(1, 1), binary= False
)

print('\n')
print("#"*20, " TERM FREQUENCY ", "#"*20)

tf = count_vector.fit_transform(corpus)
vector_tf = tf.toarray()
tf_list = []
for i in range(len(vector_tf)) : 
    tf_child = []
    for val in vector_tf[i] :
        tf_child.append(val/len(corpus[i].split()))
    tf_list.append(tf_child)

tf_df = pd.DataFrame(tf_list, columns= count_vector.get_feature_names_out())
print(tf_df)

print("#"*20, " TERM FREQUENCY ", "#"*20)
print('\n')

print('\n')
print("#"*20, " INVERSE DOCUMENT FREQUENCY ", "#"*20)

features = count_vector.get_feature_names_out()
idf_mat = []
for feature in features : 
    count = 0 
    for doc in corpus : 
        word_list = [word.lower() for word in word_tokenize(doc) if word.isalpha()]
        if feature in word_list : 
            count += 1 

    row = [np.log10((len(corpus)+1)/(count+1)) + 1] * len(corpus)
    idf_mat.append(row)

idf_df = pd.DataFrame(np.array(idf_mat).T, columns= features)
print(idf_df)

print("#"*20, " INVERSE DOCUMENT FREQUENCY ", "#"*20)
print('\n')

print("#"*20, " TF - IDF ", "#"*20)

tf_idf_cal = pd.DataFrame(np.multiply(np.array(tf_list), np.array(idf_mat).T), columns= features)
print(tf_idf_cal)

print("#"*20, " TF - IDF ", "#"*20)
print('\n')

#? tfidf vector without stop words 
word_vectorizer = TfidfVectorizer(
    input = 'content', lowercase= True, stop_words= [], norm = 'l1',
    encoding= ' utf-8', ngram_range=(1, 1), binary= False, use_idf= True, 
    smooth_idf= True
)

print("tfidf vectorizer without stop words ")
print("="*20)
X_s = word_vectorizer.fit_transform(corpus)
X_s_array = X_s.toarray()
X_s_columns = word_vectorizer.get_feature_names_out()

df_s = pd.DataFrame(X_s_array, columns = X_s_columns)
print(df_s)
print('\n')