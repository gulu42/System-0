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
import copy
import pickle
import numpy as np

from game import Agent
from qlearningAgents import ApproximateQAgent


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
        self.q_learning_agent = ApproximateQAgent(extractor = 'SimpleExtractor',numTraining=-1) #need to figure out how to pass args here
        self.q_learning_agent.setEpsilon(0)
        with open('network_weights.pkl', 'rb') as input:
            trained_weights = pickle.load(input)
            self.q_learning_agent.setWeights(trained_weights)
        with open('features.pkl', 'rb') as input:
            extractor_used = pickle.load(input)
            self.q_learning_agent.setExtractor(extractor_used)
    def getAction(self,gameState):
        return self.q_learning_agent.getAction(gameState)

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
        self.count = 0
    def getAction(self,gameState):
        # return self.system_1_model.getAction(gameState)
        #GHOST PROXIMTIY
        #######################################
        newPos = gameState.getPacmanPosition()
        # distances_to_ghosts = 1
        # proximity_to_ghosts = 0
        # for ghost_state in gameState.getGhostPositions():
        #     distance = util.manhattanDistance(newPos, ghost_state)
        #     distances_to_ghosts += distance
        #     if distance <= 2:
        #         proximity_to_ghosts += 1
        # if proximity_to_ghosts > 0:
        #     move = self.system_2_model.getAction(gameState)
        # else:
        #     move = self.system_1_model.getAction(gameState)
        #########################################
        #HEAT MAP
        #######################################
        # distances_to_ghosts = 1
        # proximity_to_ghosts = 0
        # sys2_positions = [(1,5) , (2,5) , (4,5) , (1,4), (9,5),(2,4),(1,1),(5,3)]
        # sys1_positions = [(14,5) , (16,4), (20,4),(20,2),(20,1)]
        # sys0_positions = [(16,5),(17,5),(16,3),(1,5),(2,4),(4,5),(5,5),(20,5),(1,5),(1,4)]
        # if newPos not in sys0_positions:
        #     move = self.system_2_model.getAction(gameState)
        # else:
        #     move = self.system_1_model.getAction(gameState)
        # for ghost_state in gameState.getGhostPositions():
        #     distance = util.manhattanDistance(newPos, ghost_state)
        #     distances_to_ghosts += distance
        #     if distance <= 2:
        #         proximity_to_ghosts += 1
        # if proximity_to_ghosts > 0 or newPos not in sys0_positions:
        # if proximity_to_ghosts > 0:
        #     move = self.system_2_model.getAction(gameState)
        # else:
        #     move = self.system_1_model.getAction(gameState)
        #########################################

        # self.count = self.count + 1
        # if(self.system_1_model.getAction(gameState) != self.system_2_model.getAction(gameState)):
        #     print "Made a choice"
        # print "SYSTEM1:" + self.system_1_model.getAction(gameState)
        # print "SYSTEM2:" + self.system_2_model.getAction(gameState)
        # print "Count:" + str(self.count)
        food = gameState.getNumFood()
        if food > 5:
            move = self.system_2_model.getAction(gameState)
        else:
            move = self.system_1_model.getAction(gameState)
        return move

class ProximityAgent(System0Agent):
    def __init__(self,proxi_dist=2,escape_sys=2):
        System0Agent.__init__(self)
        self.proxi_dist = float(proxi_dist)
        self.escape_sys = int(escape_sys)
        
    def getAction(self,gameState):
        # return self.system_1_model.getAction(gameState)
        newPos = gameState.getPacmanPosition()
        distances_to_ghosts = 1
        proximity_to_ghosts = 0
        for ghost_state in gameState.getGhostPositions():
            distance = util.manhattanDistance(newPos, ghost_state)
            distances_to_ghosts += distance
            if distance <= self.proxi_dist:
                proximity_to_ghosts += 1
        if proximity_to_ghosts < 1:
            if self.escape_sys == 2:
                move = self.system_1_model.getAction(gameState)
            else:
                move = self.system_2_model.getAction(gameState)
        else: #high proximity, escape
            if self.escape_sys == 2:
                move = self.system_2_model.getAction(gameState)
            else:
                move = self.system_1_model.getAction(gameState)
        return move

class RandomChoiceAgent(System0Agent):
    def __init__(self, prob_sys1):
        System0Agent.__init__(self)
        self.prob_sys1 = float(prob_sys1)
        # self.d = {1:0,2:0}

    def getAction(self,gameState):
        system_to_use = np.random.choice([1,2], p=[self.prob_sys1,1 - self.prob_sys1])
        # self.d[system_to_use] += 1
        # print self.d
        if system_to_use == 1:
            return self.system_1_model.getAction(gameState)
        else:
            return self.system_2_model.getAction(gameState)

class ProximityAndFoodAgent(System0Agent):
    def __init__(self,food_thresh=0.5):
        System0Agent.__init__(self)
        self.threshold = food_thresh
        self.total_food_num = 65.0

    def getAction(self,gameState):
        newPos = gameState.getPacmanPosition()
        distances_to_ghosts = 1
        proximity_to_ghosts = 0
        for ghost_state in gameState.getGhostPositions():
            distance = util.manhattanDistance(newPos, ghost_state)
            distances_to_ghosts += distance
            if distance <= 2:
                proximity_to_ghosts += 1

        if proximity_to_ghosts < 1:
            ratio = gameState.getNumFood()/self.total_food_num
            if (float(ratio) >= float(self.threshold)):
                move = self.system_2_model.getAction(gameState)
            else:
                move = self.system_1_model.getAction(gameState)
        else:
            move = self.system_2_model.getAction(gameState)
        return move
