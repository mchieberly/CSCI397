# Malachi Eberly
# Assignent 3: Incorporate the value function

import numpy as np

GAMMA = 0.95

def main():
    probabilityMatrix = np.array([[0, 0.5, 0.2, 0.2, 0.1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0.2, 0.2, 0, 0.6, 0, 0, 0, 0, 0, 0],
                         [0, 0.1, 0, 0, 0.8, 0, 0.1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0],
                         [0, 0, 0, 0.5, 0, 0, 0.4, 0, 0, 0, 0, 0.1],
                         [0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.2, 0.3, 0, 0],
                         [0, 0, 0, 0, 0, 0.4, 0, 0, 0.6, 0, 0, 0],
                         [0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0.5, 0, 0.2],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0.5],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0],
                         [0, 0, 0, 0, 0.3, 0, 0.3, 0, 0.2, 0, 0, 0.2],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]])
    
    # Islands 5, 7, and 10 have treasure, so they have a positive reward
    rewardMatrix = np.array([[0], [-1], [-1], [-1], [2], [-1], [2], [-1], [-1], [2], [-1], [15]])
    identity = np.identity(12)

    # Close form Bellman equation
    values = np.linalg.inv(identity - (GAMMA * probabilityMatrix)).dot(rewardMatrix)

    # Round the values
    for i in range(len(values)):
        values[i][0] = round(values[i][0], 2)
    
    i = 1
    for v in values:
        print("Value of moving to island s", i, ": ", v[0], sep = "")
        i += 1

if __name__ == "__main__":
    main()