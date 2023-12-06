import nltk 
from nltk.corpus import stopwords 

nltk.download('stopwords')
nltk.download('gutenberg')

# get the stop words 
stop_words = list(stopwords.words('english'))
words_bryant = nltk.Text(nltk.corpus.gutenberg.words('bryant-stories.txt'))
words_emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))

words_bryant = [word.lower() for word in words_bryant if word.isalpha()]
words_emma = [word.lower() for word in words_emma if word.isalpha()]

# remove stop words  
# ? taken equal length to compare 
words_bryant = [word.lower() for word in words_bryant if word not in stop_words][:15000]
words_emma = [word.lower() for word in words_emma if word not in stop_words][:15000]

TTR_bryant = len(set(words_bryant))/len(words_bryant)
TTR_emma = len(set(words_emma))/len(words_emma)

print("number of tokens, vocabulary, type-token ration (Bryant stories) --> ", len(words_bryant), len(set(words_bryant)), TTR_bryant)

print("number of tokens, vocabulary, type-token ration (Jane Austen Emma) --> ", len(words_emma), len(set(words_emma)), TTR_emma)