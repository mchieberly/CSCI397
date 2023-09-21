# Malachi Eberly
# Class for the agent

class Agent():
    def __init__(self, location, foundTreasure = 0):
        self.location = location
        self.reward = 0
        self.foundTreasure = foundTreasure

    def move(self, location):
        # Make sure that the specified location is a possible move
        self.location = location
        self.reward -= 1

    def dig(self):
        if self.location.checkTreasure == True:
            self.reward += 2
            self.foundTreasure += 1