import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets 
import torchvision.transforms as transforms
import time

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Download training data from open datasets.
training_data = datasets.CIFAR10(
    root="../data",
    train=True,
    download=False,
    transform=transform,
)

# Download test data from open datasets.
test_data = datasets.CIFAR10(
    root="../data",
    train=False,
    download=False,
    transform=transform,
)

# Create data loaders
batch_size = 64
train_dataloader = DataLoader(training_data,batch_size = batch_size)
test_dataloader = DataLoader(test_data,batch_size = batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


##### Creating Model #####
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class MLP(nn.Module):
    def __init__(self):
        super().__init__() # inherit superclass's init
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential( # do the following actions in sequence
            nn.Linear(3072,490), # 32*32*3
            nn.ReLU(),
            nn.Linear(490,10),
        )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# instantiate the model
model = MLP().to(device)

##### Train the Model #####
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

def train(dataloader,model,loss_fn,optimizer):
    size = len(dataloader.dataset)
    model.train() # Sets the module in training mode
    for batch,(X,y) in enumerate(dataloader):
        X,y = X.to(device),y.to(device)

        pred = model(X) # overload () to farward()
        loss = loss_fn(pred,y)

        # backpropagation
        optimizer.zero_grad() # set the last round's gradient to zero
        loss.backward() # *calculate gradient
        optimizer.step() # *update

        if batch % 100 == 0:
            loss,current = loss.item(),batch*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        

##### Test #####
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