import re 
from operator import itemgetter 
import nltk 
import pandas as pd

nltk.download('gutenberg')

frequency = {}
words_emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))

for word in words_emma : 
    count = frequency.get(word, 0)
    frequency[word] = count + 1

rank = 1 
column_header = ["rank", 'freq', 'freq*rank']
df = pd.DataFrame(columns = column_header)

for word, freq in reversed(sorted(frequency.items(), key = itemgetter(1))) :
    df.loc[word] = [rank, freq, rank*freq]
    rank += 1 

# this is only used for modeling the corpus 
print(df)