from transformers import BertConfig, BertModel, BertForSequenceClassification, BertTokenizer
import torch
import torch.nn as nn

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

class BERT(nn.Module):
    def __init__(self, bert_path, device):
        super(BERT, self).__init__()
        self.bert_path = bert_path
        self.config = BertConfig.from_pretrained(self.bert_path)
        self.model = BertModel.from_pretrained(self.bert_path, config=self.config)
        
    def forward(self, x):
        output = self.model(x)
        x1 = output[1]
        return x1

