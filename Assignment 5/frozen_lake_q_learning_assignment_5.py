# Malachi Eberly
# Assignment 5: Frozen Lake Q-Learning

#!/usr/bin/env python3
import gymnasium as gym
import numpy as np
from tensorboardX import SummaryWriter
from tabulate import tabulate

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME, is_slippery = True)
        self.state = self.env.reset()
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.actions = {0 : "Left", 1: "Down", 2 : "Right", 3 : "Up"}

    def sample_env(self):
        action = self.env.action_space.sample()
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        old_state = self.state
        if terminated or truncated:
            self.state = self.env.reset()
        else:
            self.state = new_state
        return old_state, action, reward, new_state

    def best_value_and_action(self, state):
        best_value, best_action = None, None
        state = state[0] if type(state) is tuple else state
        for action in range(self.env.action_space.n):
            q_value = self.q_table[state][action]
            if best_value is None or q_value > best_value:
                best_value, best_action = q_value, action
        return best_value, best_action

    def value_update(self, state, action, reward, new_state):
        best_new_value, _ = self.best_value_and_action(new_state)
        state = state[0] if type(state) is tuple else state
        new_q_value = reward + (GAMMA * best_new_value) - self.q_table[state][action]
        new_q_value = self.q_table[state][action] + (ALPHA * new_q_value)
        self.q_table[state][action] = new_q_value

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
            state = new_state
        return total_reward

    def print_values(self):
        print("Values:")
        headers = ["State"] + [f"{self.actions[action]}" for action in range(self.env.action_space.n)]
        table_data = []
        for state in range(self.env.observation_space.n):
            state_row = [f"{state}"]
            for action in range(self.env.action_space.n):
                q_value = self.q_table[state][action]
                state_row.append(f"{q_value:.3f}")
            table_data.append(state_row)
        print(tabulate(table_data, headers, tablefmt="grid"))
        print()

    def print_policy(self):
        print("Policy:")
        policy = {}
        policy_grid = [['' for _ in range(4)] for _ in range(4)]
        for state in range(self.env.observation_space.n):
            row, col = divmod(state, 4)
            _, best_action = self.best_value_and_action(state)
            policy_grid[row][col] = self.actions[best_action]
            policy[state] = self.actions[best_action]
        for row in range(4):
            for col in range(4):
                if (row == 1 and col == 1) or (row == 1 and col == 3) or (row == 2 and col == 3) or (row == 3 and col == 0):
                    policy_grid[row][col] = "HOLE"
                elif row == 3 and col == 3:
                    policy_grid[row][col] = "GOAL"
                print(f"{policy_grid[row][col]:<6}", end=" ")
            print()
        print()
        return policy

if __name__ == "__main__":
    test_env = gym.make(ENV_NAME, is_slippery = True)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        state, action, reward, new_state = agent.sample_env()
        agent.value_update(state, action, reward, new_state)

        cumulative_reward = 0.0
        for _ in range(TEST_EPISODES):
            cumulative_reward += agent.play_episode(test_env)
        cumulative_reward /= TEST_EPISODES
        writer.add_scalar("reward", cumulative_reward, iter_no)
        if cumulative_reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, cumulative_reward))
            best_reward = cumulative_reward
        if cumulative_reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()

    # Print the Q-values and extract/print the policy
    agent.print_values()
    agent.print_policy()
