from collections import defaultdict
import numpy as np
import re
import pandas as pd

class LogisticRegressionClf:
    def __init__(self, learning_rate, epochs, stopwords=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.corpus = []            # Placeholder for corpus of tweets
        self.stopwords = stopwords if stopwords else set()
        self.vocabulary = {}        # Dictionary to map each word to a unique index
        self.word_counts = defaultdict(int)  # Counts for each word in the corpus
        self.tweet_matrix = None    # Feature matrix, initialized in preprocess_data()

    def add_tweet_to_corpus(self, X_train):
        for i in range(X_train.shape[0]):
            self.corpus.append(X_train["Tweet"].iloc[i])

    def clean_tweet(self, tweet):
        tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
        tweet = tweet.lower()
        tokens = [word for word in tweet.split() if word not in self.stopwords]
        return tokens

    def tokenize_tweet(self, tweet):
        return self.clean_tweet(tweet)

    def preprocess_data(self):
        for tweet in self.corpus:
            tokenized = self.tokenize_tweet(tweet)
            for word in tokenized:
                if word not in self.vocabulary:
                    self.vocabulary[word] = len(self.vocabulary)  

                self.word_counts[word] += 1


        self.tweet_matrix = np.zeros((len(self.corpus), len(self.vocabulary)))

        for i, tweet in enumerate(self.corpus):
            tokenized = self.tokenize_tweet(tweet)
            feature_vector = np.zeros(len(self.vocabulary))
            for word in tokenized:
                if word in self.vocabulary:

                    feature_vector[self.vocabulary[word]] += 1
            self.tweet_matrix[i] = feature_vector



    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def get_X_data(self):
        return self.tweet_matrix

    def fit(self, X, y):
        num_features = X.shape[1]
        self.weights = np.zeros(num_features)
        self.bias = 0

        for epoch in range(self.epochs):
            z = np.dot(X, self.weights) + self.bias
            out = self.sigmoid(z)

            dW = (1 / X.shape[0]) * np.dot(X.T, (out - y))
            dB = (1 / X.shape[0]) * np.sum(out - y)

            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * dB


            if epoch % 1000 == 0:  # Print every 100 epochs
                cost = - (1 / X.shape[0]) * np.sum(y * np.log(out) + (1 - y) * np.log(1 - out))
                print(f"Epoch {epoch}: Cost {cost}")

    def vectorize_tweet(self, tweet):
        feature_vector = np.zeros(len(self.vocabulary))
        tokenized = self.tokenize_tweet(tweet)
        
        for word in tokenized:
            if word in self.vocabulary:
                feature_vector[self.vocabulary[word]] += 1
        return feature_vector

    def predict(self, X):  
        if isinstance(X, type(pd.DataFrame())):
            X_vectorized = np.array([self.vectorize_tweet(tweet) for tweet in X["Tweet"]])
        else:
            X_vectorized = X  
            
        print(f"X_vectorized shape after vectorization: {X_vectorized.shape}")
        print(f"Weights shape: {self.weights.shape}")
        
        if X_vectorized.shape[1] != len(self.weights):
            raise ValueError("Feature size mismatch: Check if weights are aligned with vocabulary size.")

        # Prediction calculation
        z = np.dot(X_vectorized, self.weights) + self.bias
        out = self.sigmoid(z)
        out_preds = np.where(out >= 0.5, 1, 0)
        return out_preds
