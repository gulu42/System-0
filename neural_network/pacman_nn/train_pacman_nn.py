from __future__ import print_function
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

epochs = 5
batch_size = 10

class PacmanDataset(Dataset):
    """Dataset based on pacman moves"""

    def __init__(self, csv_file_name, transform=None):
        self.df = pd.read_csv(csv_file_name)
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self,idx):
        if torch.is_tesnor(idx):
            idx = idx.tolist()

        return self.df.iloc[idx]

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
        self.final_layer = nn.LogSoftmax(dim=1)

    def forward(self,x): #define the forward pass here
        x = self.fc1(x)
        x = self.non_linearity(x)
        # --------output of first hidden layer
        x = self.fc2(x)
        x = self.non_linearity(x)
        # --------output of second hidden layer
        x = self.fc3(x)
        x = self.non_linearity(x)

        x = self.fc4(x)
        x = self.non_linearity(x)

        x = self.fc5(x)
        # x = self.s_max(x.unsqueeze(dim=0))
        # cross entropy applies softmax on its own
        # Add this to the output of the network when using it later
        # --------apply softmax and return moves probability
        x = self.final_layer(x.unsqueeze(dim=0))
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
    # net.load_state_dict(torch.load('./net.pth'))
    # print("Network:")
    # print(net)

    # move module to gpu before constructiong optimizer object
    #check availability of GPU and then use cuda to shift model there
    loss_criterion = nn.KLDivLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.5)

    df = pd.read_csv("data_shuffle.csv")
    num_rows = df.shape[0]

    inp_columns = df.drop(['Action','North','East','South','West'],axis=1).columns # the dropped column is not in-place
    num_inp = len(inp_columns)

    # Storage for target value
    target = torch.FloatTensor(4)

    for e in range(epochs):
        for index,row in df.iterrows():

            # get input and target
            inp = torch.FloatTensor(map(float, row[inp_columns]))
            target[0] = float(row["North"])
            target[1] = float(row["East"])
            target[2] = float(row["South"])
            target[3] = float(row["West"])

            # Get network output
            # net_out = net(inp).unsqueeze(dim=0) # unsqueeze to apply cross entropy loss
            net_out = net(inp) # network has a log softmax as the last layer
            # print(net_out)

            # calculate loss
            loss = loss_criterion(net_out,target)

            #update network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Loss - pass",index,": ",loss)

    torch.save(net.state_dict(), './mini_net.pth')
