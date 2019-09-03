from __future__ import print_function
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

epochs = 1

class LinearNetwork(nn.Module):
    def __init__(self): #all inits here
        super(LinearNetwork,self).__init__()
        self.fc1 = nn.Linear(3,1)

    def forward(self,x): #define the forward pass here
        x = self.fc1(x)
        return x

net = LinearNetwork()
print("Network:")
print(net)
print()

# move module to gpu before constructiong optimizer object
#check availability of GPU and then use cuda to shift model there
loss_criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(),lr = 0.1)

df = pd.read_csv("linear_train_data.csv")


for e in range(epochs):
    for index,row in df.iterrows():

        # read dataset entry
        inp = torch.tensor([row['x1'],row['x2'],row['x3']])
        target = torch.tensor([row['label']])

        #get network output
        out = net(inp)

        #calculate loss
        loss = loss_criterion(out,target)
        print("Loss - pass",index,": ",loss)

        # update network weights, backprop and weight update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("\nTesting:")

df = pd.read_csv("linear_test_data.csv")

for index,row in df.iterrows():
    # read dataset entry
    inp = torch.tensor([row['x1'],row['x2'],row['x3']])
    target = torch.tensor([row['label']])

    #get network output
    out = net(inp)

    #calculate loss
    loss = loss_criterion(out,target)
    print("Pass",index,":")
    print("Target =",target)
    print("Output =",out)
    print("Loss - pass",index,": ",loss)
    print("------------\n")
