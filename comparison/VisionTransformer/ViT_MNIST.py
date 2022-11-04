# reference:https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c

import torch
import torch.nn as nn
import torch.optim as optim

import math
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets 
from torchvision.transforms import ToTensor
import time

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="../../data",
    train=True,
    download=False,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="../../data",
    train=False,
    download=False,
    transform=ToTensor(),
)

# Create data loaders
batch_size = 64
train_dataloader = DataLoader(training_data,batch_size = batch_size)
test_dataloader = DataLoader(test_data,batch_size = batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}") # [64, 1, 28, 28]
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# patchifying and linear mapping => flattened 
# here each image is devided to 7*7 patches, each patch is 4*4
def patchify(images,n_patches):
    n,c,h,w = images.shape
    patches = torch.zeros(n,n_patches ** 2, h * w // n_patches ** 2) # [n,number of patches,patch dimension]
    patch_size = h // n_patches # length of patch
    
    for idx,image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
                patches[idx,i*n_patches+j] = patch.flatten()

    return patches

# positional encoding
def get_positional_embeddings(seq_len,d):
    result = torch.ones(seq_len,d)
    for i in range(seq_len):
        for j in range(d):
            result[i][j] = np.sin(i/(10000**(j/d))) if j % 2 == 0 else np.cos(i/(10000 ** ((j - 1)/d)))
    return result

# multi-head self attention
class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class EncoderBlock(nn.Module):
    def __init__(self,hidden_dim,n_heads,mlp_ratio=4):
        super(EncoderBlock,self).__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.msa = MSA(hidden_dim,n_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim,mlp_ratio * hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_dim,hidden_dim)
        )

    def forward(self,x):
        out = x + self.msa(self.norm1(x))
        out = out + self.mlp(self.norm2(out)) #TODO
        return out

class ViT(nn.Module):
    def __init__(self,chw,n_patches=7,n_blocks=2,hidden_dim=8,n_heads=2,out_dim=10):
        super(ViT,self).__init__()

        self.chw = chw # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        
        self.patch_size = (chw[1]/n_patches,chw[2]/n_patches)

        # linear mapper
        self.input_dim = int(chw[0]*self.patch_size[0]*self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_dim,self.hidden_dim)

        # classification token
        self.class_token = nn.Parameter(torch.rand(1,self.hidden_dim))

        # positional embedding
        # self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(self.n_patches**2+1,self.hidden_dim)))
        # self.pos_embed.requires_grad = False
        self.register_buffer('positional_embeddings', get_positional_embeddings(n_patches ** 2 + 1, hidden_dim), persistent=False)


        # encoder block
        self.blocks = nn.ModuleList([EncoderBlock(hidden_dim,n_heads) for _ in range(n_blocks)])

        # classification mlp
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim,out_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self,images):
        n,c,h,w = images.shape
        patches = patchify(images,self.n_patches).to(self.positional_embeddings.device)
        tokens = self.linear_mapper(patches)
        # add classification token
        tokens = torch.stack([torch.vstack((self.class_token,tokens[i]))for i in range(len(tokens))])
        # add positional embedding
        # pos_embed = self.pos_embed.repeat(n,1,1)
        # out = tokens + pos_embed #TODO
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)
        
        for block in self.blocks:
            out = block(out)

        # classification only
        out = out[:,0]

        # map to output dimension
        out = self.mlp(out)
        return out


model = ViT((1, 28, 28), n_patches=7, n_blocks=2, hidden_dim=8, n_heads=2, out_dim=10).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.005)
loss_fn = nn.CrossEntropyLoss()

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


epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")