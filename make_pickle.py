import gensim
from BERT import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertConfig, BertTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
import numpy
import datetime
import random

import re
import csv
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import download
import string

download("stopwords", quiet=True)

""" This class does text preprocessing. """
class PreProcessTweets:
    def __init__(self):
        self.printable = set(string.printable)
        self.stopword_list = stopwords.words("english")
            
    def processTweet(self, tweet):
        # Remove URLs and replace them with 'URL'
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' url ', tweet)
        # Remove '#' and '@'
        tweet = tweet.replace('@', ' ')
        tweet = tweet.replace('#', ' ')
        # Replace numbers with `number`
        tweet = re.sub(r'(?<=\d), [,\.]', '', tweet)
        tweet = re.sub(" \d+", " number ", tweet)
        # Break tweet into words
        tweet = word_tokenize(tweet)
        # Join tweet again
        tweet = ' '.join(tweet)
        # Remove hexadecimals
        tweet = re.sub(r'[^\x00-\x7f]',r'', tweet) 

        return tweet

BERT_MODEL = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)

method = "word2vec"
print(method)
device = "cuda:3"
path = "data/tweets.csv"

bert = BERT(BERT_MODEL, device).to(device)
for param in bert.model.parameters():
    param.requires_grad = False

# Load Google Word2Vec
word2vec = gensim.models.KeyedVectors.load_word2vec_format('word2vec_model/GoogleNews-vectors-negative300.bin', binary=True)
        
vader = SentimentIntensityAnalyzer()

# Convert tweet to BERT or Word2Vec embedding
def convert_to_embedding(sentence):
    if method=='bert':
        #bert_sent = torch.tensor(sentence).view(1, -1)
        bert_sent = bert(sentence.to(device))
        return bert_sent.tolist()

    elif method=='word2vec':
        total = numpy.zeros(300)
        words = 0
        # Averages all outputs for each word. If word has no representation in Word2Vec then an all-zero vector is used to represent it
        for w in word_tokenize(sentence):
            try:
                total += word2vec[w]
                words += 1
            except:
                pass
        if words!=0:
            total /= words
        return total.tolist()
    elif method=='vader':
        v = list(vader.polarity_scores(sentence).values())
        return v

# The dictionary that holds the tweets
d = dict()

# Reading the file
with open(path, 'r', encoding='utf-8') as file:
    tweets_reader = csv.reader(file, delimiter=';')
    # True if the file has a header line with the column names, False otherwise
    first=True
    # The text preprocessing object
    processor = PreProcessTweets()
    count = 0
    for row in tweets_reader:
        if first:
            # Header is read, go to the next row
            first = False
            continue
        # Not all rows had 9 columns, but most had, so this was added as exception prevention
        if len(row)==9:
            # Datetimes are converted into UNIX timestamp
            date_str = row[4].split()[0]
            unix = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()-datetime.date(1970, 1, 1)    
            unix = int(unix.days)       
            if unix<17136 or unix>17927:
                continue
            if count%100000 == 0:
                print(count)
            count += 1
            tweet = row[8][:280]
            tweet = processor.processTweet(tweet)
            if len(tweet.strip())==0:
                tweet = 'empty tweet'
            if method=='bert':
                tokens = tokenizer.tokenize(tweet)
                tweet = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])
                if len(tweet)<=40:
                    tweet.extend([0]*(40-len(tweet)))
                else:
                    tweet = tweet[:39]+[102]
            if unix in d:
                # Update entry of timestamp with a new preprocessed tweet
                d[unix].append(tweet)
            else:
                # Create new entry for the timestamp
                d[unix] = [tweet]
print('loaded in ram')
K = len(list(d.keys()))
k = 0
for i in d:
    print(100*k/K)
    k += 1
    if method=='bert':
        dt = torch.tensor(d[i])
        l = []
        loader = DataLoader(dt, batch_size=32)
        for x in loader:
            l.extend(convert_to_embedding(x))
        d[i] = l
    else:
        d[i] = [convert_to_embedding(j) for j in d[i]]
print('calculated embeddings')

pickle.dump(d, open(method+".p", "wb"), protocol=4)

