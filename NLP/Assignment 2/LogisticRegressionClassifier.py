## OOP!
from collections import defaultdict
import numpy as np
import re


class LogisticRegressionClf:
    def __init__(self, learning_rate, epochs, stopwords = None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.corpus = set()
        self.stopwords = stopwords if stopwords else set()
        self.word_counts = {}

    def clean_tweet(self, tweet):
        tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
        tweet = tweet.lower()
        tokens = [word for word in tweet if word not in self.stopwords]
        return tokens


    def tokenize_tweet(self, tweet):
        tweet = self.clean_tweet(tweet)
        return tweet
    
    def sigmoid(self):
        pass

    def preprocess_data(self):
        for tweet in self.corpus:
            tokenized = self.tokenize_tweet(tweet)
            if tokenized not in self.word_counts.keys():
                self.word_counts[tokenized] = 1
            else:
                self.word_counts[tokenized] += 1

    def init_features(self):
        return np.zeros()
