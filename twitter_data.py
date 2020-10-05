import random
import datetime
import time

import numpy
import torch
import pickle

""" This class generates embeddings for training the textual part of the proposed model. 
It outputs Word2Vec, BERT or VADER embeddings, according to the 'method' input variable. """
class Load_twitter_data():
    def __init__(self, num_tweets, method=None, device='cpu'):
        self.data_twitter = dict()
        if method != None:
            self.data_twitter = pickle.load(open("pickle/"+method+".p", "rb"))
        self.num_tweets = num_tweets
        self.method = method
        self.device = device   

    # This method returns a dataloader for a certain day
    def get_embeddings(self, date):
        random.seed(time.time())
        # Search in dictionary for tweets from day
        unix = int((date-datetime.date(1970, 1, 1)).days)
        day_tweets = random.sample(self.data_twitter[unix], min(self.num_tweets, len(self.data_twitter[unix])))
        day_tweets = torch.tensor(day_tweets)
        # Average tweet embeddings
        day_tweets = torch.mean(day_tweets, 0)

        return day_tweets

