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

import sys
from util import manhattanDistance
from game import Directions
import random, util
import torch
import torch.nn as nn
import copy

from game import Agent

sys.path.insert(1,'../neural_network/pacman_nn')
from train_pacman_nn import PacmanNetwork

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

class System1Agent(Agent): #system 1 is capable of gameplay on its own
    """
    Code modelling actions of system 1 comes here
    """

    def __init__(self):
        self.network = PacmanNetwork()
        self.network.load_state_dict(torch.load('../neural_network/pacman_nn/net.pth'))
        self.s_max = nn.Softmax(dim=1)
    def getAction(self,gameState):
        rows = list()

        # Extract game state to pass to networs ----- start
        # grid_values = gameState.getWalls().shallowCopy()
        grid_values = [[0 for i in range(7)] for j in range(20)]

        # fill walls as -3, -4 for now, will add 1 to it for empty cells
        for i in range(20):
            for j in range(7):
                grid_values[i][j] = -4*gameState.getWalls()[i][j]

        # fill food cells as +3, empty cells as +1
        for i in range(20):
            for j in range(7):
                grid_values[i][j] = 2*gameState.getFood()[i][j] + gameState.getWalls()[i][j] + 1
        print(gameState.getWalls())
        legalMoves = gameState.getLegalActions()
        # fill pacman position as 0
        x,y = gameState.getPacmanPosition()
        grid_values[x][y] = 0

        # fill ghost position as -10
        for i in range(2):
            x,y = map(int,gameState.getGhostPositions()[i])
            grid_values[x][y] = -10

        # fill capsule positions as +10
        for i in range(len(gameState.getCapsules())):
            x,y = gameState.getCapsules()[i]
            grid_values[x][y] = +10

        # fill rows and columns to add to dataset
        for i in range(20):
            for j in range(7):
                rows.append(grid_values[i][j])

        # Extract game state to pass to networs ----- end

        # print(grid_values)

        inp = torch.FloatTensor(map(float, rows))
        out = self.network(inp)
        out = self.s_max(out.unsqueeze(dim=0)).squeeze(dim=0)
        print("network prob:",out)
        legalMoves = gameState.getLegalActions()
        print("legal moves:",legalMoves)
        max_move = 'Stop'
        max_prob = 0
        print out
        if 'North' in legalMoves and out[0] >= max_prob:
            max_move = 'North'
            max_prob = out[0]
        elif 'East' in legalMoves and out[1] >= max_prob:
            max_move = 'East'
            max_prob = out[1]
        elif 'South' in legalMoves and out[2] >= max_prob:
            max_move = 'South'
            max_prob = out[2]
        elif 'West' in legalMoves and out[3] >= max_prob:
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
