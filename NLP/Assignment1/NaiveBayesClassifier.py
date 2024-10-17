## OOP!
from collections import defaultdict
import numpy as np
import re


class NaiveBayesClf:
    def __init__(self, stopwords = None):
        self.corpus = set()
        self.word_frequencies = {}
        self.class_prior_probability = {}
        self.class_word_counts = {}
        self.stopwords = stopwords if stopwords else set()

    def clean_tweet(self, tweet):
        tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
        tweet = tweet.lower()

        tokens = [word for word in tweet if word not in self.stopwords]
        return tokens

    def tokenize(self, tweet):
        tweet = self.clean_tweet(tweet)
        return tweet

    
    def fit(self, X_train, y_train):
        self.classes = np.array([0,1])
        self.class_prior_probability = {cl: 0 for cl in self.classes}
        self.word_frequencies = {cl: defaultdict(int) for cl in self.classes}
        self.class_word_counts = {c: 0 for c in self.classes}
        

        for tweet, label in zip(X_train, y_train):
            tokens = self.tokenize(tweet)
            self.class_prior_probability[label] += 1
            self.class_word_counts[label] += len(tokens)

            for token in tokens:
                self.corpus.add(token)
                self.word_frequencies[label][token] += 1

        total_tweets = len(y_train)
        for clas in self.classes:
            self.class_prior_probability[clas] /= total_tweets

    def likelihood(self, token, label):
        return (self.word_frequencies[label][token] + 1) / (self.class_word_counts[label] + len(self.corpus))


    def predict(self, X_test):
        predictions = []        
        for tweet in X_test:
            tokens = self.tokenize(tweet)
            log_probabilities = {cl: np.log(self.class_prior_probability[cl]) for cl in self.classes}

            for c in self.classes:
                for token in tokens:
                    log_probabilities[c] += np.log(self.likelihood(token, c))
            
            predictions.append(max(log_probabilities, key = log_probabilities.get))

        return predictions