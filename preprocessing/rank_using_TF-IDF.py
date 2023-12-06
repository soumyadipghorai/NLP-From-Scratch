import nltk
import numpy as np
from numpy.linalg import norm 
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer

# downloadables if not already downladed 
nltk.download('stopwords')
nltk.download('gutenberg')

class Preprocessor :
    def convertListToText(self, word_list: list[str]) -> str: 
        output = ""
        for word in word_list : 
            output += word + ' '
        return output[:-1]
    
    def checkCosineSimilarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        return np.dot(vector1, vector2)/(norm(vector1)*norm(vector2))

# get the stop words 
stop_words = list(stopwords.words('english'))
words_bryant = nltk.Text(nltk.corpus.gutenberg.words('bryant-stories.txt'))
words_emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))
words_persuasion = nltk.Text(nltk.corpus.gutenberg.words('austen-persuasion.txt'))
words_sense = nltk.Text(nltk.corpus.gutenberg.words('austen-sense.txt'))

words_bryant = [word.lower() for word in words_bryant if word.isalpha()]
words_emma = [word.lower() for word in words_emma if word.isalpha()]
words_persuasion = [word.lower() for word in words_persuasion if word.isalpha()]
words_sense = [word.lower() for word in words_sense if word.isalpha()]

# remove stop words  
# ? taken equal length to compare 
words_bryant = [word.lower() for word in words_bryant if word not in stop_words]
words_emma = [word.lower() for word in words_emma if word not in stop_words]
words_persuasion = [word.lower() for word in words_persuasion if word not in stop_words]
words_sense = [word.lower() for word in words_sense if word not in stop_words]

preprocessor = Preprocessor()
words_bryant_text = preprocessor.convertListToText(words_bryant)
words_emma_text = preprocessor.convertListToText(words_emma)
words_persuasion_text = preprocessor.convertListToText(words_persuasion)
words_sense_text = preprocessor.convertListToText(words_sense)

vectorizer = TfidfVectorizer(
    input = 'content', encoding = 'utf-8', lowercase = True, 
    decode_error = 'ignore'
)

corpus = [words_bryant_text, words_emma_text, words_persuasion_text, words_sense_text]
names = ["bryant", "emma", "persuasion", "sense"]

X = vectorizer.fit_transform(corpus)
tfidf_matrix = X.toarray()

query = input("enter query : ").strip()
tfidf_query = np.ravel(vectorizer.transform(query.split(' ')).toarray())

cosine_score = {}
for i in range(tfidf_matrix.shape[0]) :
    similarity = preprocessor.checkCosineSimilarity(tfidf_matrix[i], tfidf_query)
    cosine_score[names[i]] = similarity 

for doc, score in cosine_score.items() : 
    print(f"{doc} --> {score}")