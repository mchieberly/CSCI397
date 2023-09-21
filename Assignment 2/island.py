# Malachi Eberly
# Class for an island state

class Island():
    def __init__(self, name, treasure):
        self.name = name
        self.treasure = treasure

    def checkIsland(self):
        if self.treasure == "treasure!":
            return True
        return False
    
    