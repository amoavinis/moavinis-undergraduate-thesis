import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

""" This class instantiates a Fully Connected layer that goes on top of the LSTM and textual parts. """
class Dense(nn.Module):
    def __init__(self, only_financial, only_textual, lstm_dim, text_dim, text_lstm_dim, mode='bert', device='cpu'):
        super(Dense, self).__init__()
        self.mode = mode

        self.device = device
        self.fc1 = nn.Linear(6*lstm_dim, 16)        
        self.fc2 = nn.Linear(16, 3)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.bert_fc1 = nn.Linear(768, 300)
        self.bert_fc2 = nn.Linear(300, 200)
        self.bert_fc3 = nn.Linear(200, 100)
        self.bert_fc4 = nn.Linear(100, 3)   

        self.only_x1 = only_financial
        self.only_x2 = only_textual

        self.w2v = nn.Linear(300, 300)
        self.w2v2 = nn.Linear(300, 300)
        self.w2v3 = nn.Linear(300, 300)
        self.w2v4 = nn.Linear(300, 3)
        
        self.vader_fc1 = nn.Linear(4, 300)
        self.vader_fc2 = nn.Linear(300, 300)
        self.vader_fc3 = nn.Linear(300, 300)
        self.vader_fc4 = nn.Linear(300, 300)
        self.vader_fc5 = nn.Linear(300, 300)
        self.vader_fc6 = nn.Linear(300, 300)
        self.vader_fc7 = nn.Linear(300, 3)

        self.final_fc1 = nn.Linear(316, 3)
        self.final_fc2 = nn.Linear(16, 3)

    def forward(self, x1, x2):
        if self.only_x1 and not self.only_x2:
            x1 = self.fc1(x1.view(x1.shape[0], -1))
            x1 = self.relu(x1)
            x1 = self.fc2(x1)
            x1 = self.tanh(x1)
            return x1
        elif not self.only_x1 and self.only_x2 and self.mode=='bert':
            x2 = self.bert_fc1(x2)
            x2 = self.relu(x2)
            x2 = self.bert_fc2(x2)
            x2 = self.relu(x2)
            x2 = self.bert_fc3(x2)
            x2 = self.relu(x2)
            x2 = self.bert_fc4(x2)
            x2 = self.tanh(x2)
            return x2
        elif not self.only_x1 and self.only_x2 and self.mode=='word2vec':
            x2 = self.w2v(x2)
            x2 = self.relu(x2)
            x2 = self.w2v2(x2)
            x2 = self.relu(x2)
            x2 = self.w2v3(x2)
            x2 = self.relu(x2)
            x2 = self.w2v4(x2)
            x2 = self.tanh(x2)
            return x2
        elif not self.only_x1 and self.only_x2 and self.mode=='vader':
            x2 = self.vader_fc1(x2)
            x2 = self.relu(x2)
            x2 = self.vader_fc2(x2)
            x2 = self.relu(x2)
            x2 = self.vader_fc3(x2)
            x2 = self.relu(x2)
            x2 = self.vader_fc4(x2)
            x2 = self.relu(x2)
            x2 = self.vader_fc5(x2)
            x2 = self.relu(x2)
            x2 = self.vader_fc6(x2)
            x2 = self.relu(x2)
            x2 = self.vader_fc7(x2)
            x2 = self.tanh(x2)
            return x2
        elif self.only_x1 and self.only_x2 and self.mode=='bert':
            x1 = self.fc1(x1.view(x1.shape[0], -1))
            x1 = self.relu(x1)
            x2 = self.bert_fc1(x2)
            x2 = self.relu(x2)
            x2 = self.bert_fc2(x2)
            x2 = self.relu(x2)
            x2 = self.bert_fc3(x2)
            x2 = self.relu(x2)
            X = torch.cat((x1, x2), dim=1)
            x = self.final_fc(X)
            x = self.tanh(x2)
            return x
        elif self.only_x1 and self.only_x2 and self.mode=='word2vec':
            x1 = self.fc1(x1.view(x1.shape[0], -1))
            x1 = self.relu(x1)
            x2 = self.w2v(x2)
            x2 = self.relu(x2)
            x2 = self.w2v2(x2)
            x2 = self.relu(x2)
            x2 = self.w2v3(x2)
            x2 = self.relu(x2)
            x = torch.cat((x1, x2), dim=1)
            x = self.final_fc1(x)
            #x = self.relu(x)
            #x = self.final_fc2(x)
            x = self.tanh(x)
            return x
        elif self.only_x1 and self.only_x2 and self.mode=='vader':
            x1 = self.fc1(x1.view(x1.shape[0], -1))
            x1 = self.relu(x1)
            x2 = self.vader_fc1(x2)
            x2 = self.relu(x2)
            x2 = self.vader_fc2(x2)
            x2 = self.relu(x2)
            x2 = self.vader_fc3(x2)
            x2 = self.relu(x2)
            x2 = self.vader_fc4(x2)
            x2 = self.relu(x2)
            x2 = self.vader_fc5(x2)
            x2 = self.relu(x2)
            x2 = self.vader_fc6(x2)
            x2 = self.relu(x2)
            x = torch.cat((x1, x2), dim=1)
            x = self.final_fc1(x)
            #x = self.relu(x)
            #x = self.final_fc2(x)
            x = self.tanh(x)
            return x

