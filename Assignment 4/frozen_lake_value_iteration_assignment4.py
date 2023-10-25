# Malachi Eberly
# Assignment 4: Frozen Lake Value Iteration

import gymnasium as gym
from collections import Counter, defaultdict
import numpy as np

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20
SEED = 42

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME, desc=None, map_name="4x4", is_slippery=True)
        self.state = 0
        self.actions = {0 : "Left", 1 : "Down", 2 : "Right", 3 : "Up"}
        self.rewards = defaultdict(lambda: defaultdict(lambda: 0))
        self.transits = defaultdict(lambda: Counter())
        self.nStates = self.env.observation_space.n
        self.nActions = self.env.action_space.n
        self.values = np.zeros(self.nStates)

    def update_transits_rewards(self, state, action, new_state, reward):
        key = (state, action)
        self.rewards[key][new_state] += reward
        self.transits[key][new_state] += 1

    def play_n_random_steps(self, count):
        self.env.reset(seed=SEED)
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, terminated, truncated, info = self.env.step(action)
            self.update_transits_rewards(self.state, action, new_state, reward)
            if terminated or truncated:
                self.env.reset(seed=SEED)
                self.state = 0
                continue
            self.state = new_state

    def print_value_table(self):
        print("Value Table:", end = "")
        for state in range(self.env.observation_space.n):
            if state % 4 == 0:
                print("\n")
            print(f"{self.values[state]:.3f}  ", end="")
        print("\n")

    def extract_policy(self):
        policy = []
        for state in range(self.nStates):
            best_action = self.select_action(state)
            policy.append(best_action)
        return policy
    
    def print_policy(self, policy):
        print("Policy:")
        printed_policy = ""
        for i, value in enumerate(policy):
            if i % 4 == 0:
                printed_policy += '\n'
            printed_policy += " {:>5} ".format(self.actions[value])
        print(printed_policy)

    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        sum_counts = sum(target_counts.values())
        action_value = 0
        for pair in target_counts:
            proportion = target_counts[pair] / sum_counts
            reward = self.rewards[(state, action)][pair]      
            action_value += proportion * (reward + GAMMA * self.values[pair])
        return action_value

    def select_action(self, state):
        best_action = float("-inf")
        best_action_value = float("-inf")
        for action in range(self.nActions):
            action_value = self.calc_action_value(state, action)
            if action_value > best_action_value:
                best_action_value = action_value
                best_action = action
        return best_action

    def play_episode(self):
        self.env.reset(seed=SEED)
        total_reward = 0
        state = 0
        count = 0
        while True:
            reward = 0
            action = self.select_action(state)
            new_state, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            count += 1
            if done or truncated:
                self.env.reset(seed=SEED)
                break
            state = new_state
        return total_reward


    def value_iteration(self):
        for state in range(self.nStates):
            state_values = [self.calc_action_value(state, action) for action in range(self.nActions)]
            self.values[state] = max(state_values)
            
if __name__ == "__main__":
    agent = Agent()

    iter_no = 0
    best_reward = float('-inf')
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = sum([agent.play_episode() for _ in range(TEST_EPISODES)]) / TEST_EPISODES
        
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward

        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            agent.print_value_table()
            policy = agent.extract_policy()
            agent.print_policy(policy)
            break