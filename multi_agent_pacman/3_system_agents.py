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

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    """Calculating distance to the closest food pellet"""
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newFoodList = newFood.asList()
    min_food_distance = -1
    for food in newFoodList:
        distance = util.manhattanDistance(newPos, food)
        if min_food_distance >= distance or min_food_distance == -1:
            min_food_distance = distance

    """Calculating the distances from pacman to the ghosts. Also, checking for the proximity of the ghosts (at distance of 1) around pacman."""
    distances_to_ghosts = 1
    proximity_to_ghosts = 0
    for ghost_state in currentGameState.getGhostPositions():
        distance = util.manhattanDistance(newPos, ghost_state)
        distances_to_ghosts += distance
        if distance <= 1:
            proximity_to_ghosts += 1

    """Obtaining the number of capsules available"""
    newCapsule = currentGameState.getCapsules()
    numberOfCapsules = len(newCapsule)

    """Combination of the above calculated metrics."""
    return currentGameState.getScore() + (1 / float(min_food_distance)) - (1 / float(distances_to_ghosts)) - proximity_to_ghosts - numberOfCapsules


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
        self.s_max = nn.Softmax(dim=1)
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
        out = self.s_max(out.unsqueeze(dim=0)).squeeze(dim=0)
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

    def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def getAction(self, gameState):
        # print("Game state type:",type(gameState))
        # print("Game state:",gameState)
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agent, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  # return the utility in case the defined depth is reached or the game is won/lost.
                return self.evaluationFunction(gameState)
            if agent == 0:  # maximizing for pacman
                return max(expectimax(1, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
            else:  # performing expectimax action for ghosts/chance nodes.
                nextAgent = agent + 1  # calculate the next agent and increase depth accordingly.
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1
                return sum(expectimax(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent)) / float(len(gameState.getLegalActions(agent)))

        """Performing maximizing task for the root node i.e. pacman"""
        maximum = float("-inf")
        action = Directions.WEST
        for agentState in gameState.getLegalActions(0):
            utility = expectimax(1, 0, gameState.generateSuccessor(0, agentState))
            if utility > maximum or maximum == float("-inf"):
                maximum = utility
                action = agentState
        return action

class System0Agent(Agent):
    def __init__(self,sys1 = Agent(),sys2 = Agent()):
        self.system_1_model = System1Agent()
        self.system_2_model = System2Agent()

    def getAction(self,gameState):
        return random.choice([self.system_1_model.getAction(gameState),self.system_2_model.getAction(gameState)])
