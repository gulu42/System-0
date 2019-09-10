from __future__ import print_function
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

epochs = 5
batch_size = 5

class PacmanDataset(Dataset):
    """Dataset based on pacman moves"""

    def __init__(self, csv_file_name, transform=None):
        self.df = pd.read_csv(csv_file_name)
        self.df = self.df.drop("Action", axis = 1) # since actions are there in th epacman dataset
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return torch.tensor(self.df.iloc[idx])

class PacmanNetwork(nn.Module):
    def __init__(self): #all inits here
        super(PacmanNetwork,self).__init__()
        # self.non_linearity = nn.ReLU()
        self.non_linearity = nn.SELU()
        self.s_max = nn.Softmax(dim=1)
        self.fc1 = nn.Linear(140,110)
        self.fc2 = nn.Linear(110,80)
        self.fc3 = nn.Linear(80,50)
        self.fc4 = nn.Linear(50,20)
        self.fc5 = nn.Linear(20,4)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self,x): #define the forward pass here
        x = self.fc1(x)
        x = self.non_linearity(x)
        x = self.dropout(x)
        # --------output of first hidden layer
        x = self.fc2(x)
        x = self.non_linearity(x)
        x = self.dropout(x)
        # --------output of second hidden layer
        x = self.fc3(x)
        x = self.non_linearity(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.non_linearity(x)

        x = self.fc5(x)
        return x

def get_class_given_action(action):
    # North, East, South, West, Stop

    if action == "North":
        return 0
    elif action == "East":
        return 1
    elif action == "South":
        return 2
    elif action == "West":
        return 3

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        net = PacmanNetwork().cuda()
    else:
        net = PacmanNetwork()

    loss_criterion = nn.KLDivLoss()
    log_smax = nn.LogSoftmax(dim=1) # KL-Divergence expects a log softmax of the neural network output
    optimizer = optim.Adam(net.parameters(),lr = 0.5)

    dataset = PacmanDataset("data.csv")
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle = True, num_workers=8)

    for e in range(epochs):
        # for index,row in df.iterrows():
        for i,data in enumerate(trainloader,0):

            # get input and target
            inp = data[:,:-4] # for each row in th ebatch, pick everything except the last 4 columns (target values)
            target = data[:,-4:]

            # Get network output
            net_out = net(inp) # network has a log softmax as the last layer

            # calculate loss
            loss = loss_criterion(log_smax(net_out.unsqueeze(dim=0)),target) #the log softmax needs a one layer of padding, the network output doesn't have this

            #update network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Loss - epoch",e,": ",loss)

    torch.save(net.state_dict(), './net.pth')
