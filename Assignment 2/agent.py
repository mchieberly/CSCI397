# Malachi Eberly
# Class for the agent

import island

class Agent():
    def __init__(self, location, foundTreasure = 0):
        self.location = location
        self.reward = 0
        self.foundTreasure = foundTreasure

    def move(self, location):
        self.location = location
        self.reward -= 1

    def dig(self):
        if self.location.checkTreasure == True:
            self.reward += 2
            self.foundTreasure += 1