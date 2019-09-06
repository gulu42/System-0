from __future__ import print_function
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

epochs = 5

class PacmanNetwork(nn.Module):
    def __init__(self): #all inits here
        super(PacmanNetwork,self).__init__()
        # self.non_linearity = nn.ReLU()
        self.non_linearity = nn.SELU()
        self.s_max = nn.Softmax(dim=1)
        self.fc1 = nn.Linear(140,60)
        self.fc2 = nn.Linear(60,20)
        self.fc3 = nn.Linear(20,5)

    def forward(self,x): #define the forward pass here
        x = self.fc1(x)
        x = self.non_linearity(x)
        # --------output of first hidden layer
        x = self.fc2(x)
        x = self.non_linearity(x)
        # --------output of second hidden layer
        x = self.fc3(x)
        # x = self.s_max(x.unsqueeze(dim=0))
        # cross entropy applies softmax on its own
        # Add this to the output of the network when using it later
        # --------apply softmax and return moves probability
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
    else:
        return 4

net = PacmanNetwork()
# net.load_state_dict(torch.load('./net.pth'))
print("Network:")
print(net)

# move module to gpu before constructiong optimizer object
#check availability of GPU and then use cuda to shift model there
loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.1)

df = pd.read_csv("data.csv")

inp_columns = df.drop('Action',axis=1).columns # the dropped column is not in-place
num_inp = len(inp_columns)

# Storage for target value
target = torch.LongTensor(1)

for e in range(epochs):
    for index,row in df.iterrows():

        # get input and target
        inp = torch.FloatTensor(map(float, row[inp_columns]))
        target[0] = get_class_given_action(row["Action"])

        # Get network output
        net_out = net(inp).unsqueeze(dim=0) # unsqueeze to apply cross entropy loss
        # print(net_out)

        # calculate loss
        loss = loss_criterion(net_out,target)
        print("Loss - pass",index,": ",loss)

        #update network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(net.state_dict(), './net.pth')
