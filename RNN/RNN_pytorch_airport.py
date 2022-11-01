# reference:https://github.com/L1aoXingyu/code-of-learn-deep-learning-with-pytorch/blob/master/chapter5_RNN/time-series/lstm-time-series.ipynb
# using LSTM
# introduction to LSTM:https://zhuanlan.zhihu.com/p/104475016
# introduction to simplest RNN:https://zhuanlan.zhihu.com/p/30844905

from ast import increment_lineno
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn 
from torch.autograd import Variable

##### Data Preprocessing #####
data = pd.read_csv('../data/airport_data.csv',usecols=[1])
# plt.plot(data)
# plt.show()

data = data.dropna() #remove NA 
dataset = data.values
dataset = dataset.astype('float32')
# normalize the data to 0-1
scalar = np.max(dataset) - np.min(dataset)
dataset = list(map(lambda x : x/scalar,dataset))

# use the last two months' data to predict this month's data
def create_dataset(dataset,look_back=2): 
    dataX,dataY=[],[] 
    for i in range(len(dataset)-look_back):
        dataX.append(dataset[i:(i+look_back)])
        dataY.append(dataset[i+look_back])
    return np.array(dataX),np.array(dataY)

data_X,data_Y = create_dataset(dataset)

# 70% of the dataset is for training 
train_size = int(len(data_X)*0.7)
train_X = data_X[:train_size]
test_X = data_X[train_size:]
train_Y = data_Y[:train_size]
test_Y = data_Y[train_size:]

#print(test_X)
train_X = train_X.reshape(-1,1,2) # (seq_len,batch,feature) todo
train_Y = train_Y.reshape(-1,1,1)
test_X = test_X.reshape(-1,1,2)
train_X = torch.from_numpy(train_X)
train_Y = torch.from_numpy(train_Y)
test_X = torch.from_numpy(test_X)
#print(test_X)

##### Define the Model #####
class LSTM_net(nn.Module):
    def __init__(self,input_size,hidden_size,output_size=1,num_layers=2):
        super(LSTM_net,self).__init__() # if use "super().__init__()" here,the result would be bad
        self.rnn = nn.LSTM(input_size,hidden_size,num_layers)
        self.fc = nn.Linear(hidden_size,output_size) #todo

    def forward(self,x):
        x,_ = self.rnn(x)
        s,b,h = x.shape
        x = x.view(s*b,h)
        x = self.fc(x)
        x = x.view(s,b,-1)
        return x

model = LSTM_net(2,4)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)

##### Train the Model #####
for batch in range(1000):
    var_X = Variable(train_X)
    var_Y = Variable(train_Y)

    pred = model(var_X)
    loss = loss_fn(pred,var_Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # if (batch+1) % 100 == 0:
    #         print('Epoch: {}, Loss: {:.5f}'.format(batch+1, loss.data[0]))
    if batch % 100 == 0:
        loss = loss.item()
        print(f"loss: {loss:>7f}  [{batch+100:>5d}/{1000}]")


##### Test #####
model = model.eval()
data_X = data_X.reshape(-1, 1, 2)
data_X = torch.from_numpy(data_X)
var_data = Variable(data_X)
pred_test = model(var_data)

pred_test = pred_test.view(-1).data.numpy()

plt.plot(pred_test, 'r', label='prediction')
plt.plot(dataset, 'b', label='real')
plt.legend(loc='best')
plt.show()