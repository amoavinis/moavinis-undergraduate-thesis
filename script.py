# Imports
from LSTM import *
from FC import *
from twitter_data import *
from stock_data import *
from utilities import *
from pandas import DataFrame, concat, date_range
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AdamW, get_linear_schedule_with_warmup
import datetime
import time
from matplotlib import pyplot as plt

# Setting the seed for reproducibility
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# Ticker to get financial info for
ticker = 'BTC-EUR'
print(ticker)

# Setting the CUDA device to run the script on
device = 'cuda:1'
print(device)

# Batch size
BATCH_SIZE = 32

# Use financial data and textual data
only_financial = True
only_textual = True

# Financial data will be used from the last 7 days, Twitter data from the last day
stock_step = 7

# Data between these dates will be used
starting_date = '2017-1-1'
ending_date = '2019-1-1'
print(starting_date)
print(ending_date)

# Calculate number of days
start = datetime.datetime.strptime(starting_date, '%Y-%m-%d').date()
end = datetime.datetime.strptime(ending_date, '%Y-%m-%d').date()
range_len = end - start
range_len = range_len.days + 1

# For each day
ranges = []
for i in range(range_len):
    # Calculate starting datetime and ending datetime to search in tweets database
    date = start + datetime.timedelta(days=i)
    ranges.append(date-datetime.timedelta(days=1))

# Financial data loader
financial_data = Load_financial_data(ticker,
                 from_date='2017-1-1',
                 to_date='2019-1-1',
                 step=stock_step,
                 split=0.9,
                 batch_size=BATCH_SIZE 
)
train_loader, valid_loader, closing_train, closing_valid,\
              y_train, y_valid = financial_data.get_loader()

# Choose algorithm, BERT or Word2Vec
method = 'word2vec'
if method=='bert':
    text_dim = 768
elif method=='word2vec':
    text_dim = 300
elif method=='vader':
    text_dim = 4
else:
    text_dim = 1
f = open(str(method)+'_'+str(int(time.time()))+'.txt', 'w')

# How many tweets to use per day
NUM_TWEETS = 1000

print(str(method))
f.write(str(method)+'\n')
if method!='word2vec' and method!='bert' and method!='vader' and method!=None:
    print("Specify a text processing model.")
    exit()

lstm_dim = 120

# Initializing models
stock_model = LSTM(input_dim=stock_step, hidden_dim=lstm_dim, output_dim=6,
                   num_layers=1, device=device).to(device)
fc = Dense(only_financial, only_textual, lstm_dim, text_dim, 120, mode=method, device=device).to(device)

print("Getting tweets into RAM...")
# Initialize Tweets data loader
if only_textual:
    tweets_loader = Load_twitter_data(NUM_TWEETS, method=method, device=device)
else:
    tweets_loader = Load_twitter_data(NUM_TWEETS, method=None, device=device)
print("OK.")

# Mean Square Error Loss Function
loss_fn = nn.CrossEntropyLoss()

# Calculating and printing the naive loss for the validation dataset
tanh = nn.Tanh()
scores = [0, 0, 0]
for y in y_valid:
    scores[y.item()]+=1
loss_naive = 0.0
for y in y_valid:
    y = y.view(1)
    yhat = torch.tensor([[0, 0, 0]]).float()
    yhat[0][scores.index(max(scores))] = 1
    loss_naive+=loss_fn(tanh(yhat), y)
naive_loss = loss_naive.item()/len(y_valid)
naive_acc = 100*max(scores)/sum(scores)
print("Naive loss: "+str(naive_loss))
f.write(str(naive_loss)+'\n')
print("Naive accuracy: "+str(naive_acc)[:6])
f.write(str(naive_acc)[:6]+'\n')

# This is used to create the loss graphs
training_loss_history = []
validation_loss_history = []

# Training part

# Number of epochs
n_epochs = 1000

# Initializing optimizers and scheduler
optimizer1 = optim.RMSprop(stock_model.parameters(), lr=0.01)
optimizer2 = optim.Adam(fc.parameters(), lr=0.001)

for epoch in range(n_epochs):
    stock_model.train()
    fc.train()
    t = time.time()
    train_loss = 0.0
    train_acc = 0.0
    # `pointer` starts at 0 and increases for every batch member for all batches
    pointer = 0
    print("Epoch ", epoch+1, '/', n_epochs, sep='')
    f.write("Epoch "+str(epoch+1)+'/'+str(n_epochs)+'\n')

    for x_batch, y_batch in train_loader:
        # Getting financial data
        x_batch = x_batch.view(x_batch.shape[0], x_batch.shape[1], x_batch.shape[2]).to(device)
        y_batch = y_batch.view(y_batch.shape[0]).to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        
        text_outputs = []
        if only_textual:
            for i in range(x_batch.size(0)):
                tweets = tweets_loader.get_embeddings(ranges[pointer])
                pointer += 1
                if len(tweets)==0:
                    tweets = torch.zeros(text_dim).to(device)
                text_outputs.append(tweets.tolist())
        text_outputs = torch.tensor(text_outputs).float().to(device)
        
        # Through Financial LSTM model
        fin_lstm_out = stock_model(x_batch)

        # Pass LSTM and textual output into final layer
        yhat = fc(fin_lstm_out.to(device), text_outputs.to(device))

        # Calculate loss and do the backwards pass
        loss = loss_fn(yhat, y_batch)
        loss.backward()

        # Run optimization steps
        optimizer1.step()
        optimizer2.step()
        
        # Calculate training loss and accuracy
        train_loss += loss.item()*x_batch.shape[0]
        top_p, top_class = yhat.topk(1, dim=1)
        equals = top_class == y_batch.view(*top_class.shape)
        train_acc += torch.mean(equals.type(torch.FloatTensor)).item()*x_batch.shape[0]

    training_str = "Training accuracy: "+"{0:.2f}".format(100*train_acc/len(train_loader.dataset)) + " Training loss: "+str(train_loss/len(train_loader.dataset))
    print(training_str)
    f.write(training_str+'\n')
    time_str = "Seconds per epoch: " + str(round(time.time()-t, 2))
    print(time_str)
    f.write(time_str+'\n')
    training_loss_history.append(train_loss/len(train_loader.dataset))

    # Validation part
    valid_preds = []
    valid_true = []
    total_valid = 0
    correct_valid = 0
    valid_loss = 0.0
    valid_acc = 0.0

    stock_model.eval()
    fc.eval()

    # We continue getting datetimes from the point we stopped, that being the index of the end of the training loader
    pointer = len(train_loader.dataset)
    with torch.no_grad():
        for x_batch, y_batch in valid_loader:
            # Getting financial data
            x_batch = x_batch.view(x_batch.shape[0], x_batch.shape[1], x_batch.shape[2]).to(device)
            y_batch = y_batch.view(y_batch.shape[0]).to(device)
            
            text_outputs = []
            if only_textual:
                for i in range(x_batch.size(0)):
                    tweets = tweets_loader.get_embeddings(ranges[pointer])
                    pointer += 1
                    if len(tweets)==0:
                        tweets = torch.zeros(text_dim).to(device)
                    text_outputs.append(tweets.tolist())
            text_outputs = torch.tensor(text_outputs).float().to(device)

            # Through Financial LSTM model
            fin_lstm_out = stock_model(x_batch)

            # Pass LSTM and textual output into final layer
            yhat = fc(fin_lstm_out.to(device), text_outputs.to(device))

            # Calculate loss
            loss = loss_fn(yhat, y_batch)
            
            # Calculate validation loss and accuracy
            valid_loss += loss.item()*x_batch.shape[0]
            top_p, top_class = yhat.topk(1, dim=1)
            valid_preds.extend(top_class.view(-1).tolist())
            valid_true.extend(y_batch.view(-1).tolist())
        
            equals = top_class == y_batch.view(*top_class.shape)
            valid_acc += torch.mean(equals.type(torch.FloatTensor)).item()*x_batch.shape[0]
        c_matrix = print_confusion_matrix(valid_true, valid_preds)
        print(c_matrix)
        f.write(str(c_matrix)+'\n')
            
    valid_str = "Validation accuracy: "+"{0:.2f}".format(100*valid_acc/len(valid_loader.dataset))+" Validation loss: "+str(valid_loss/len(valid_loader.dataset))
    print(valid_str)
    f.write(valid_str+'\n')
    validation_loss_history.append(valid_loss/len(valid_loader.dataset))
f.close()

plt.plot(training_loss_history, color='k')
plt.plot(validation_loss_history, color='r', linestyle='dashed')
plt.plot([naive_loss]*n_epochs, color='g', linestyle='dashdot')
plt.show()
