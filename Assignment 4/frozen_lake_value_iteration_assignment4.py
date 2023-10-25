# # Malachi Eberly
# # Assignment 4: Frozen Lake Value Iteration

# import gymnasium as gym
# from collections import Counter, defaultdict
# import numpy as np

# ENV_NAME = "FrozenLake-v1"
# GAMMA = 0.9
# TEST_EPISODES = 20
# SEED = 42

# class Agent:
#     def __init__(self):
#         self.env = gym.make(ENV_NAME, desc=None, map_name="4x4", is_slippery=True)
#         self.state = 0
#         self.actions = {0 : "Left", 1 : "Down", 2 : "Right", 3 : "Up"}
#         self.rewards = defaultdict(lambda: defaultdict(lambda: 0))
#         self.transits = defaultdict(lambda: Counter())
#         self.nStates = self.env.observation_space.n
#         self.nActions = self.env.action_space.n
#         self.values = np.zeros(self.nStates)

#     def update_transits_rewards(self, state, action, new_state, reward):
#         key = (state, action)
#         self.rewards[key][new_state] += reward
#         self.transits[key][new_state] += 1

#     def play_n_random_steps(self, count):
#         self.state = 0
#         self.env.reset(seed=SEED)
#         for _ in range(count):
#             action = self.env.action_space.sample()
#             expected_reward = 0
#             for _ in range(self.nActions):
#                 new_state, reward, terminated, truncated, _ = self.env.step(action)
#                 expected_reward += (reward + GAMMA * self.values[new_state]) / self.nActions
#                 if terminated or truncated:
#                     self.env.reset(seed=SEED)
#                     self.state = 0
#             self.update_transits_rewards(self.state, action, new_state, expected_reward)
#             self.state = new_state

#     def print_value_table(self):
#         print("Value Table:", end = "")
#         for state in range(self.env.observation_space.n):
#             if state % 4 == 0:
#                 print("\n")
#             print("{0:>7.3f}".format(self.values[state]), end="")
#         print("\n")

#     def extract_policy(self):
#         policy = []
#         for state in range(self.nStates):
#             best_action = self.select_action(state)
#             policy.append(best_action)
#         return policy
    
#     def print_policy(self, policy):
#         print("Policy:", end = "")
#         printed_policy = ""
#         for i, value in enumerate(policy):
#             if i % 4 == 0:
#                 printed_policy += "\n\n"
#             if i == 15:
#                 printed_policy += "{:>7}".format("GOAL")
#             elif i in [5, 7, 11, 12]:
#                 printed_policy += "{:>7}".format("HOLE")
#             else:
#                 printed_policy += "{:>7}".format(self.actions[value])
#         print(printed_policy)

#     def calc_action_value(self, state, action):
#         target_counts = self.transits[(state, action)]
#         sum_counts = sum(target_counts.values())
#         action_value = 0
#         for new_state, count in target_counts.items():
#             proportion = count / sum(target_counts.values())
#             reward = self.rewards[(state, action)][new_state]
#             action_value += proportion * (reward + GAMMA * self.values[new_state])
#         return action_value

#     def select_action(self, state):
#         best_action = 0 # This is important
#         best_action_value = 0
#         for action in range(self.nActions):
#             action_value = self.calc_action_value(state, action)
#             if action_value > best_action_value:
#                 best_action_value = action_value
#                 best_action = action
#         return best_action

#     def play_episode(self):
#         self.env.reset(seed=SEED)
#         total_reward = 0
#         self.state = 0
#         while True:
#             reward = 0
#             action = self.select_action(self.state)
#             new_state, reward, done, truncated, info = self.env.step(action)
#             total_reward += reward
#             if done or truncated:
#                 self.env.reset(seed=SEED)
#                 break
#             self.state = new_state
#         return total_reward

#     def value_iteration(self):
#         for state in range(self.nStates):
#             state_values = [self.calc_action_value(state, action) for action in range(self.nActions)]
#             self.values[state] = max(state_values)
            
# if __name__ == "__main__":
#     agent = Agent()

#     iter_no = 0
#     best_reward = 0
#     while True:
#         iter_no += 1
#         agent.play_n_random_steps(100)
#         agent.value_iteration()

#         rewards = 0
#         for _ in range(TEST_EPISODES):
#             rewards += agent.play_episode()
#         reward = rewards / TEST_EPISODES
        
#         if reward > best_reward:
#             print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
#             best_reward = reward

#         if reward > 0.80:
#             print("Solved in %d iterations!" % iter_no)
#             agent.print_value_table()
#             policy = agent.extract_policy()
#             agent.print_policy(policy)
#             break

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
        self.env = gym.make(ENV_NAME, desc=None, map_name="4x4", is_slippery=False)
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
        self.state = 0
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
            print("{0:>7.3f}".format(self.values[state]), end="")
        print("\n")

    def extract_policy(self):
        policy = []
        for state in range(self.nStates):
            best_action = self.select_action(state)
            policy.append(best_action)
        return policy
    
    def print_policy(self, policy):
        print("Policy:", end = "")
        printed_policy = ""
        for i, value in enumerate(policy):
            if i % 4 == 0:
                printed_policy += "\n\n"
            if i == 15:
                printed_policy += "{:>7}".format("GOAL")
            elif i in [5, 7, 11, 12]:
                printed_policy += "{:>7}".format("HOLE")
            else:
                printed_policy += "{:>7}".format(self.actions[value])
        print(printed_policy)

    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        sum_counts = sum(target_counts.values())
        action_value = 0
        for ntransits in target_counts:
            proportion = target_counts[ntransits] / sum_counts
            reward = self.rewards[(state, action)][ntransits]
            action_value += proportion * (reward + GAMMA * self.values[ntransits])
        return action_value

    def select_action(self, state):
        best_action = 0 # This is important
        best_action_value = 0
        for action in range(self.nActions):
            action_value = self.calc_action_value(state, action)
            if action_value > best_action_value:
                best_action_value = action_value
                best_action = action
        return best_action

    def play_episode(self):
        self.env.reset(seed=SEED)
        total_reward = 0
        self.state = 0
        while True:
            reward = 0
            action = self.select_action(self.state)
            new_state, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                self.env.reset(seed=SEED)
                break
            self.state = new_state
        return total_reward

    def value_iteration(self):
        for state in range(self.nStates):
            state_values = [self.calc_action_value(state, action) for action in range(self.nActions)]
            self.values[state] = max(state_values)
            
if __name__ == "__main__":
    agent = Agent()

    iter_no = 0
    best_reward = 0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        rewards = 0
        for _ in range(TEST_EPISODES):
            rewards += agent.play_episode()
        reward = rewards / TEST_EPISODES
        
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward

        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            agent.print_value_table()
            policy = agent.extract_policy()
            agent.print_policy(policy)
            break