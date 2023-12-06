import nltk 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def create_doc_from_path(path) :
    text_doc = open(path, 'r')
    doc = [nltk.word_tokenize(line.lower()) for line in text_doc.readlines()]
    words = []
    for line in doc : 
        for word in line : 
            if word.isalpha() : 
                words.append(word)
    return words

doc1 = create_doc_from_path('text_docs/doc1.txt')
doc2 = create_doc_from_path('text_docs/doc2.txt')
doc3 = create_doc_from_path('text_docs/doc3.txt')

corpus = [doc1, doc2, doc3]
corpus = [" ".join(doc) for doc in corpus]

vocabulary = set()
for doc in corpus : 
    for word in doc.split(" ") : 
        vocabulary.add(word)

vocabulary = sorted(vocabulary)


vectorizer = CountVectorizer(
    vocabulary=vocabulary, lowercase= True, encoding='utf-8'
)

X = vectorizer.fit_transform(corpus)
X = X.toarray()

print("="*30)
print('numpy respresentation')
print(X)
print("shape of array --> ", X.shape)
print("length of vocabulary --> ", len(vocabulary))

print("="*30)
print('dataframe respresentation')
df = pd.DataFrame(X, columns= vectorizer.get_feature_names_out())
print(df)