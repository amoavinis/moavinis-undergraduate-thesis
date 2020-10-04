from pandas import DataFrame, concat, date_range
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sklearn.preprocessing
import yfinance as yf
import numpy as np

""" This class returns a dataloader for the financial data for a ticker. """
class Load_financial_data():
    def __init__(self, stock, from_date, to_date, split=0.8, step=7, batch_size=1):
        self.stock = stock
        self.from_date = from_date
        self.to_date = to_date
        self.split = split
        self.step = step
        self.batch_size = batch_size
        
    # Create a differenced series
    def percentile_changes(self, data):
        changes = []
        for i in range(len(data)-1):
            p = data[i+1]/data[i]
            p -= 1
            p *= 100
            changes.append(p)
        return np.array(changes)

    def inverse_percentile_changes(self, x, dy):
        values = []
        values.append(x)
        for d in dy:
            p = (1+d/100.0)*values[-1]
            values.append(p)
        return values

    # Frame a sequence as a supervised learning problem
    def timeseries_to_supervised(self, data, lag=1):
        df = DataFrame(data)
        columns = [df.shift(i) for i in range(lag, 0, -1)]
        columns.append(df)
        df = concat(columns, axis=1)
        df.fillna(0, inplace=True)
        return df
    # Split into train and valid data
    def train_valid(self, data):
        data = np.array(self.percentile_changes(data))
        supervised = self.timeseries_to_supervised(data, self.step).values
        train = supervised[:self.split]
        valid = supervised[self.split:]
        return train[:,:-1], valid[:,:-1], train[:,-1], valid[:,-1]
    # Create dataloader for the financial data
    def get_loader(self):
        self.df = yf.download(self.stock,
                 start=self.from_date,
                 end=self.to_date,
                 progress=False)
        # Custom dataset
        class MyDataset(Dataset):
            def __init__(self, datas, labels):
                self.datas = datas
                self.labels = labels

            def __getitem__(self, index):
                data, target = self.datas[index], self.labels[index] 
                return data, target

            def __len__(self):
                return len(self.datas)
            
        self.opening = list(self.df['Open'].values)
        self.high = list(self.df['High'].values)
        self.low = list(self.df['Low'].values)
        self.diff = list(self.df['High'].values-self.df['Low'].values)
        self.closing = list(self.df['Close'].values)
        self.adj = list(self.df['Adj Close'].values)
        self.volume = list(self.df['Volume'].values)
        
        self.split = int(self.split*len(self.opening))-1
        self.n = len(self.opening)-1
        
        thr = 0.6

        opening_train, opening_valid, _, _ = self.train_valid(self.opening)
        high_train, high_valid, _, _ = self.train_valid(self.high)
        low_train, low_valid, _, _ = self.train_valid(self.low)
        diff_train, diff_valid, _, _ = self.train_valid(self.diff)
        closing_train, closing_valid, self.y_train, self.y_valid = self.train_valid(self.closing)
        adj_train, adj_valid, _, _ = self.train_valid(self.adj)
        vol_train, vol_valid, _, _ = self.train_valid(self.volume)
        
        self.closing_train = self.closing[:self.split+1]
        self.closing_valid = self.closing[self.split+1:]
        
        self.x_train_tensor = torch.cat(
            (
                torch.tensor(opening_train).float().view(-1, 1, self.step),
                torch.tensor(high_train).float().view(-1, 1, self.step),
                torch.tensor(low_train).float().view(-1, 1, self.step),
                torch.tensor(closing_train).float().view(-1, 1, self.step),
                torch.tensor(adj_train).float().view(-1, 1, self.step),
                torch.tensor(vol_train).float().view(-1, 1, self.step)
            ),
            1
        )
        
        for i in range(len(self.y_train)):
            if self.y_train[i].item()<-thr:
                self.y_train[i] = 0
            elif self.y_train[i].item()>thr:
                self.y_train[i] = 2
            else:
                self.y_train[i] = 1
        self.y_train = torch.from_numpy(self.y_train).clone().long() 
        self.y_train_tensor = self.y_train.clone()
        
        self.x_valid_tensor = torch.cat(
            (
                torch.tensor(opening_valid).float().view(-1, 1, self.step),
                torch.tensor(high_valid).float().view(-1, 1, self.step),
                torch.tensor(low_valid).float().view(-1, 1, self.step),
                torch.tensor(closing_valid).float().view(-1, 1, self.step),
                torch.tensor(adj_valid).float().view(-1, 1, self.step),
                torch.tensor(vol_valid).float().view(-1, 1, self.step)
            ),
            1
        )
        for i in range(len(self.y_valid)):
            if self.y_valid[i].item()<-thr:
                self.y_valid[i] = 0
            elif self.y_valid[i].item()>thr:
                self.y_valid[i] = 2
            else:
                self.y_valid[i] = 1   
        self.y_valid = torch.from_numpy(self.y_valid).clone().long()     
        self.y_valid_tensor = self.y_valid.clone()
        
        self.train_loader = DataLoader(MyDataset(self.x_train_tensor,
                                                 self.y_train_tensor),
                                       batch_size=self.batch_size, shuffle=False,
                                       num_workers=0,
                                       pin_memory=True)
        self.valid_loader = DataLoader(MyDataset(self.x_valid_tensor,
                                                self.y_valid_tensor),
                                      batch_size=self.batch_size, shuffle=False,
                                      num_workers=0,
                                      pin_memory=True)

        return self.train_loader, self.valid_loader,\
               self.closing_train, self.closing_valid,\
               self.y_train, self.y_valid
