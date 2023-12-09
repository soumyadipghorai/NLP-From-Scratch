import re 
from operator import itemgetter 
import nltk 
import pandas as pd
import matplotlib.pyplot as plt 
from nltk.corpus import stopwords
nltk.download('stopwords')

nltk.download('gutenberg')

stop_words = set(stopwords.words('english'))
frequency = {}
words_emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))

# number of unique tokens = KT^b ==> T total tokens 
# b = 0.49 
# 30 <= k <= 100 

corpus = [word for word in words_emma if word.isalpha() and word not in stop_words]

corpus_len = len(corpus)
unique_word = len(set(corpus))

b = 0.49 
predictions = []
for k in range(30, 101) :
    predicted_unique_words = k * pow(corpus_len, b)
    predictions.append(predicted_unique_words)

plt.plot([i for i in range(30, 101)], predictions, color = 'blue', label = 'predicted for different k');
plt.plot([i for i in range(30, 101)], [unique_word]*len(predictions), color = 'red', linestyle = "dashed", label = 'actual unique words');
plt.xlabel("k");
plt.ylabel("predicted corpus size");
plt.title("k vs corpus size in heap's law")
plt.yscale("log");
plt.legend();
plt.show();