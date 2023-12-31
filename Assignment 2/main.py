# Malachi Eberly
# Assignent 2: Create an agent to navigate an MDP using probabilities

import random

import island
import agent

GAMMA = 0.95

def main():
    # Setup
    s1 = island.Island("Port", "none", 0)
    s2 = island.Island("Happy", "none", 1)
    s3 = island.Island("Shadow", "none", 2)
    s4 = island.Island("Sandy", "none", 3)
    s5 = island.Island("Barren", "treasure!", 4)
    s6 = island.Island("Cozy", "none", 5)
    s7 = island.Island("Starry", "treasure!", 6)
    s8 = island.Island("Rocky", "none", 7)
    s9 = island.Island("Scorched", "none", 8)
    s10 = island.Island("Sacred", "treasure!", 9)
    s11 = island.Island("Kraken", "none", 10)
    s12 = island.Island("Destination", "none", 11)

    locations = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12]
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
    rewards = []
    episode = 1

    # Run simulation
    while episode <= 10:
        traveler = agent.Agent(s1)
        history = []
        timestep = 0
        currentTreasure = 0
        while(True):
            reward = 0
            action = ""
            if random.random() <= 0.1:
                action = "dig"
                traveler.dig()
                if traveler.foundTreasure > currentTreasure:
                    currentTreasure = traveler.foundTreasure
                    reward = 2
                    rewards.append(round((GAMMA ** timestep) * 2, 2))
            else:
                action = "move"
                nextIsland = random.choices(locations, probabilityMatrix[traveler.location.index])[0]
                traveler.move(nextIsland)
                reward = -1
                rewards.append(round((GAMMA ** timestep) * -1, 2))
                
            history.append((reward, traveler.location.name + " Island", action))
            timestep += 1
            if traveler.location == s12:
                simEnd = "reached terminal state"
                break
            if timestep == 25:
                simEnd = "reached 25 time-steps"
                break

        totalReward = 0
        for i in range(len(rewards)):
            totalReward += rewards[i]
        if currentTreasure == 3:
            totalReward += (GAMMA ** timestep) * 15

        print("\nEpisode,", episode, "finished:", simEnd)
        print("Episode history:", history, sep = "\n")
        print("Cummulative reward:", round(totalReward, 2))
        episode += 1

    print("\nAll 10 episodes finished")


if __name__ == "__main__":
    main()