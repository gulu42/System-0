# line 587, pacman.py - the process for loading an agent has been defined
# 3_system_agents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import torch
import torch.nn as nn

from game import Agent

# ideally import this network
class PacmanNetwork(nn.Module):
    def __init__(self): #all inits here
        super(PacmanNetwork,self).__init__()
        # self.non_linearity = nn.ReLU()
        self.non_linearity = nn.SELU()
        self.s_max = nn.Softmax(dim=1)
        self.fc1 = nn.Linear(152,60)
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

class System1Agent(Agent): #system 1 is capable of gameplay on its own
    """
    Code modelling actions of system 1 comes here
    """

    def __init__(self):
        self.network = PacmanNetwork()
        self.network.load_state_dict(torch.load('../neural_network/pacman_nn/net.pth'))
        self.values = 0
        self.indicies = 0
    def getAction(self,gameState):
        columns = list()
        rows = list()
        rows = [gameState.getPacmanPosition()[0],gameState.getPacmanPosition()[1], \
        gameState.getGhostPositions()[0][0],gameState.getGhostPositions()[0][1], \
        gameState.getGhostPositions()[1][0],gameState.getGhostPositions()[1][1] , \
        gameState.getNumFood(), gameState.getScore()]

        if len(gameState.getCapsules()) == 2:
            for i in range(len(gameState.getCapsules())):
                rows.append(gameState.getCapsules()[i][0])
                rows.append(gameState.getCapsules()[i][1])
        elif len(gameState.getCapsules()) == 1:
            for i in range(len(gameState.getCapsules())):
                rows.append(gameState.getCapsules()[i][0])
                rows.append(gameState.getCapsules()[i][1])
            rows.append(-1)
            rows.append(-1)
        else:
            for i in range(4):
                rows.append(-1)

        for i in range(20):
            for j in range(7):
                gameState.getWalls()[i][j] = -1*gameState.getWalls()[i][j]
        for i in range(20):
            for j in range(7):
                rows.append(gameState.getFood()[i][j] + gameState.getWalls()[i][j])
                columns.append("Grid" + str(i) + "_" + str(j))
        inp = torch.FloatTensor(map(float, rows))
        out = self.network(inp) 
        # self.values, self.indices = out.max(0)
        print self.values
        print self.indicies
        legalMoves = gameState.getLegalActions()
        print(legalMoves)
        max_move = 'Stop'
        max_prob = 0
        print out
        if 'North' in legalMoves and out[0] > max_prob:
            max_move = 'North'
            max_prob = out[0]
        elif 'East' in legalMoves and out[1] > max_prob:
            max_move = 'East'
            max_prob = out[1]
        elif 'South' in legalMoves and out[2] > max_prob:
            max_move = 'South'
            max_prob = out[2]
        elif 'West' in legalMoves and out[3] > max_prob:
            max_move = 'West'
            max_prob = out[3]
        else:
            return "Stop"
        return max_move


class System2Agent(Agent): #system 2 is capable of gameplay on its own
    """
    Code modelling actions of system 2 comes here
    """

    def getAction(self,gameState):
        legalMoves = gameState.getLegalActions()
        return random.choice(legalMoves) #arb for now

class System0Agent(Agent):
    def __init__(self,sys1 = Agent(),sys2 = Agent()):
        self.system_1_model = sys1
        self.system_2_model = sys2

    def getAction(self,gameState):
        return random.choice([self.system_1_model.getAction(gameState),self.system_2_model.getAction(gameState)])
