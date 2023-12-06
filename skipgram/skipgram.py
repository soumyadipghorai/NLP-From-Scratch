from nltk.corpus.reader import PlaintextCorpusReader
import numpy as np 

class skip_gram : 
    def __init__(
        self, corpus_dir, vocabulary_size, training_samples, context_window_size = 5, 
        word_embedding_size = 70, epochs = 400, eta = 0.01
    ) :
        self.eta = eta 
        self.epochs = epochs 
        self.vocabulary_size = vocabulary_size
        self.context_window_size = context_window_size
        self.word_embedding_size = word_embedding_size
        self.setup_corpus = PlaintextCorpusReader(corpus_dir, ".*")
        # random input weights 
        self.embedding_weights = np.random.uniform(-0.9, 0.9, (
            self.vocabulary_size, self.word_embedding_size
        )) 
        self.context_weights = np.random.uniform(-0.9, 0.9, (
            self.word_embedding_size, self.vocabulary_size
        ))
        # including X and y 
        self.training_samples = training_samples 
        self.error = []

    def softmax(self, U) : 
        pass 

    def forward_pass(self, X) : 
        H = np.dot(self.embedding_weights.T, X)
        U = np.dot(self.context_weights.T, H)
        y_hat = self.softmax(U)
        return y_hat, H, U
    
    def back_propogation(self, X, H, E) : 
        delta_context_weights = np.outer(H, E)
        delta_embedding_weights = np.outer(
            X, np.dot(self.context_weights, E.T)
        )
        self.context_weights = self.context_weights - (self.eta * delta_context_weights)
        self.embedding_weights = self.embedding_weights - (self.eta * delta_embedding_weights)

    def train(self) : 
        for i in range(self.epochs) :
            for target_word, context_words in np.array(self.training_samples) : 
                y_hat, H, U = self.forward_pass(target_word)
                # error for every context word 
                EI = np.sum([np.subtract(y_hat, word) for word in context_words], axis = 0)
                self.back_propogation(target_word, H, EI)
                self.error[i] = -np.sum([U[word.index(1)] for word in context_words]) + len(context_words) * np.log(np.sum(np.exp(U)))

# after creating word vector you can use them to find the similarity b/w different words in the corpus 