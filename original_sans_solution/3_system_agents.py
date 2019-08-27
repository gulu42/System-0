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

from game import Agent

class System1Agent(Agent): #system 1 is capable of gameplay on its own
    """
    Code modelling actions of system 1 comes here
    """

    def getAction(self,gameState):
        legalMoves = gameState.getLegalActions()
        return random.choice(legalMoves) #arb for now


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
