import nltk 
import pandas as pd 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')

stop_words_list = list(stopwords.words('english')) 

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

"""
* here you can pass either the text content in a corpus 
* or you can pass in file names 
* or you can pass in file objects with a read method. 
* text can be transformed to lowercase and can pass custom stop_words list. 
* also set binary = True for only binary matrix representation
* ngram_range will give the (min_range, max_range) ==> (1, 2) means unigram and bigrams
* max_features = n will select top n features based on term frequency accross the corpus  
"""

print('\n'*2)
print("#"*20, " CORPUS ", "#"*20)
for text in corpus :
    print(text)
print("#"*20, " CORPUS ", "#"*20)
print('\n'*2)

#? count vector without stop words 
word_vectorizer = CountVectorizer(
    input = 'content', lowercase= True, stop_words= [], 
    encoding= ' utf-8', ngram_range=(1, 1),  #unigrams only
    binary= False # set to true will give only 1 for non-zero values
)

print("count vectorizer without stop words ")
print("="*20)
X_s = word_vectorizer.fit_transform(corpus)
X_s_array = X_s.toarray()
X_s_columns = word_vectorizer.get_feature_names_out()

df_s = pd.DataFrame(X_s_array, columns = X_s_columns)
print(df_s)
print('\n'*2)

#? count vector with stop words 
word_vectorizer = CountVectorizer(
    input = 'content', lowercase= True, stop_words= stop_words_list, 
    encoding= ' utf-8', ngram_range=(1, 1),  #unigrams only
    binary= False # set to true will give only 1 for non-zero values
)

print("count vectorizer with stop words ")
print("="*20)
X = word_vectorizer.fit_transform(corpus)
X_array = X.toarray()
X_columns = word_vectorizer.get_feature_names_out()

df = pd.DataFrame(X_array, columns = X_columns)
print(df)
print('\n'*2)

#? binary vector 
word_vectorizer = CountVectorizer(
    input = 'content', lowercase= True, stop_words= stop_words_list, 
    encoding= ' utf-8', ngram_range=(1, 1),  #unigrams only
    binary= True # set to true will give only 1 for non-zero values
)

print("binary vectorizer with stop words ")
print("="*20)
X1 = word_vectorizer.fit_transform(corpus)
X1_array = X1.toarray()
X1_columns = word_vectorizer.get_feature_names_out()

df1 = pd.DataFrame(X1_array, columns = X1_columns)
print(df1)
print('\n'*2)

#? count vectorizer with ngrams
word_vectorizer = CountVectorizer(
    input = 'content', lowercase= True, stop_words= [], 
    encoding= ' utf-8', ngram_range=(1, 3),  #unigrams and bigrams
    binary= False # set to true will give only 1 for non-zero values
)

print("binary vectorizer with stop words ")
print("="*20)
X2 = word_vectorizer.fit_transform(corpus)
X2_array = X2.toarray()
X2_columns = word_vectorizer.get_feature_names_out()

df2 = pd.DataFrame(X2_array, columns = X2_columns)
print(df2)
print('\n'*2)

#? count vectorizer with ngrams with max_features 
word_vectorizer = CountVectorizer(
    input = 'content', lowercase= True, stop_words= [], 
    encoding= ' utf-8', ngram_range=(1, 3),  #unigrams and bigrams
    binary= False, # set to true will give only 1 for non-zero values
    max_features= 10
)

print("binary vectorizer with stop words ")
print("="*20)
X3 = word_vectorizer.fit_transform(corpus)
X3_array = X3.toarray()
X3_columns = word_vectorizer.get_feature_names_out()

df3 = pd.DataFrame(X3_array, columns = X3_columns)
print(df3)
print(df3.shape)
print('\n'*2)