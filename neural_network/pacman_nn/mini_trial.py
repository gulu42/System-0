from __future__ import print_function
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

batch_size = 5

class PacmanDataset(Dataset):
    """Dataset based on pacman moves"""

    def __init__(self, csv_file_name, transform=None):
        self.df = pd.read_csv(csv_file_name)
        self.df = self.df.drop("Action", axis = 1)
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return torch.tensor(self.df.iloc[idx])

dataset = PacmanDataset("data_shuffle.csv")
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle = True)

for i,data in enumerate(trainloader,0):
    print(data[:,:-4])
