# Malachi Eberly
# Class for an island state

class Island():
    def __init__(self, name, treasure, index):
        self.name = name
        self.treasure = treasure
        self.index = index

    def checkIsland(self):
        if self.treasure == "treasure!":
            self.treasure = "none"
            return True
        return False
    
