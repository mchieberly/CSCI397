# Malachi Eberly
# Assignent 2: Create an agent to navigate an MDP using probabilities

import island
import agent

def setup():
    s1 = island.Island("Port", "none", [s2, s3, s4, s5])
    s2 = island.Island("Happy", "none", [s3, s4, s6])
    s3 = island.Island("Shadow", "none", [s2, s5, s7])
    s4 = island.Island("Sandy", "none", [s8])
    s5 = island.Island("Barren", "treasure", [s4, s7, s12])
    s6 = island.Island("Cozy", "none", [s7, s8, s9, s10])
    s7 = island.Island("Starry", "treasure", [s6, s9])
    s8 = island.Island("Rocky", "none", [s4, s10])
    s9 = island.Island("Scorched", "none", [s10, s12])
    s10 = island.Island("Sacred", "treasure", [s11])
    s11 = island.Island("Kraken", "none", [s5, s7, s9, s12])
    s12 = island.Island("Destination", "none", [s12])

    traveler = agent.Agent(s1)

    probabilityMatrix = [[0, 0.5, 0.2, 0.2, 0.1, 0, 0, 0, 0, 0, 0, 0],
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
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]]

def main():
    setup()


if __name__ == "__main__":
    main()