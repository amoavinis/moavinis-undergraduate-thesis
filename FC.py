import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

""" This class instantiates a Fully Connected layer that goes on top of the LSTM and textual parts. """
class Dense(nn.Module):
    def __init__(self, do_financial, do_textual, lstm_dim, mode='word2vec', device='cpu'):
        super(Dense, self).__init__()

        self.mode = mode
        self.device = device
        self.do_financial = do_financial
        self.do_textual = do_textual

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.fc_financial_1 = nn.Linear(6*lstm_dim, 16)        
        self.fc_financial_2 = nn.Linear(16, 3)

        self.bert_fc1 = nn.Linear(768, 300)
        self.bert_fc2 = nn.Linear(300, 300)
        self.bert_fc3 = nn.Linear(300, 300)
        self.bert_fc4 = nn.Linear(300, 3)   

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

        self.final_fc = nn.Linear(316, 3)
        self.final_fc_ensemble = nn.Linear(616, 3)

    def forward(self, x1, x2, x3=None):
        if self.do_financial and not self.do_textual:
            # LSTM part
            x1 = self.fc_financial_1(x1.view(x1.shape[0], -1))
            x1 = self.relu(x1)
            x1 = self.fc_financial_2(x1)
            x1 = self.tanh(x1)
            return x1
        elif not self.do_financial and self.do_textual and self.mode=='bert':
            # BERT part
            x2 = self.bert_fc1(x2)
            x2 = self.relu(x2)
            x2 = self.bert_fc2(x2)
            x2 = self.relu(x2)
            x2 = self.bert_fc3(x2)
            x2 = self.relu(x2)
            x2 = self.bert_fc4(x2)
            x2 = self.tanh(x2)
            return x2
        elif not self.do_financial and self.do_textual and self.mode=='word2vec':
            # Word2Vec part
            x2 = self.w2v(x2)
            x2 = self.relu(x2)
            x2 = self.w2v2(x2)
            x2 = self.relu(x2)
            x2 = self.w2v3(x2)
            x2 = self.relu(x2)
            x2 = self.w2v4(x2)
            x2 = self.tanh(x2)
            return x2
        elif not self.do_financial and self.do_textual and self.mode=='vader':
            # VADER part
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
        elif self.do_financial and self.do_textual and self.mode=='bert':
            # LSTM part
            x1 = self.fc_financial_1(x1.view(x1.shape[0], -1))
            x1 = self.relu(x1)
            # BERT part
            x2 = self.bert_fc1(x2)
            x2 = self.relu(x2)
            x2 = self.bert_fc2(x2)
            x2 = self.relu(x2)
            x2 = self.bert_fc3(x2)
            x2 = self.relu(x2)
            # Concatenate and feed into final FC
            x = torch.cat((x1, x2), dim=1)
            x = self.final_fc(x)
            x = self.tanh(x2)
            return x
        elif self.do_financial and self.do_textual and self.mode=='word2vec':
            # LSTM part
            x1 = self.fc_financial_1(x1.view(x1.shape[0], -1))
            x1 = self.relu(x1)
            # Word2Vec part
            x2 = self.w2v(x2)
            x2 = self.relu(x2)
            x2 = self.w2v2(x2)
            x2 = self.relu(x2)
            x2 = self.w2v3(x2)
            x2 = self.relu(x2)
            # Concatenate and feed into final FC
            x = torch.cat((x1, x2), dim=1)
            x = self.final_fc(x)
            x = self.tanh(x)
            return x
        elif self.do_financial and self.do_textual and self.mode=='vader':
            # LSTM part
            x1 = self.fc_financial_1(x1.view(x1.shape[0], -1))
            x1 = self.relu(x1)
            # VADER part
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
            # Concatenate and feed into final FC
            x = torch.cat((x1, x2), dim=1)
            x = self.final_fc(x)
            x = self.tanh(x)
            return x
        elif self.mode=='ensemble':
            # LSTM part
            x1 = self.fc_financial_1(x1.view(x1.shape[0], -1))
            x1 = self.relu(x1)
            # VADER part
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
            # Word2Vec part
            x3 = self.w2v(x3)
            x3 = self.relu(x3)
            x3 = self.w2v2(x3)
            x3 = self.relu(x3)
            x3 = self.w2v3(x3)
            x3 = self.relu(x3)
            # Concatenate and feed into final FC
            x = torch.cat((x1, x2, x3), dim=1)
            x = self.final_fc_ensemble(x)
            x = self.tanh(x)
            return x

