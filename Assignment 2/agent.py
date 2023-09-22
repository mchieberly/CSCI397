# Malachi Eberly
# Class for the agent

class Agent():
    def __init__(self, location, foundTreasure = 0):
        self.location = location
        self.foundTreasure = foundTreasure

    def move(self, location):
        self.location = location

    def dig(self):
        if self.location.checkIsland():
            self.foundTreasure += 1