import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import time

# download the dataset MNIST
train_data = datasets.MNIST(
    root = '../data',
    train = 'True',
    transform = ToTensor(),
    download = True,
)

test_data = datasets.MNIST(
    root = '../data',
    train = False,
    transform = ToTensor(),
)
 
batch_size = 64
train_dataloader = DataLoader(train_data,batch_size = batch_size)
test_dataloader = DataLoader(test_data,batch_size = batch_size)

for X,y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Check available device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device") 

# Define the model
input_size = 28
seq_len = 28
hidden_size = 250
num_layers = 1
output_size = 10
class RNN_net(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(RNN_net,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size,output_size)
        
    def forward(self,x):
        x = x.squeeze()
        # set initial hidden and cell states to 0
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out,_ = self.lstm(x,(h_0,c_0))
        out = self.fc(out[:,-1,:]) # use the last one of the result(also remove seq_len)
        return out
    
model = RNN_net(input_size,hidden_size,num_layers).to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

# Train the data
def train(dataloader,model,loss_fn,optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch,(X,y) in enumerate(dataloader):
        X,y = X.to(device),y.to(device)
        pred = model(X)
        loss = loss_fn(pred,y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss,current = loss.item(),batch*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        
# Test
def test(dataloader,model,loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # Sets the module in evaluation mode
    test_loss,correct = 0,0
    with torch.no_grad(): # reduce memory consumption for computations
        for X,y in dataloader:
            X,y = X.to(device),y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
start_time = time.time()
epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')