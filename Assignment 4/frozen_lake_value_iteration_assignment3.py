# Malachi Eberly
# Assignment 4: Frozen Lake Value Iteration

import gymnasium as gym
import numpy as np

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20
SEED = 42

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME, desc=None, map_name="4x4", is_slippery=False)
        self.state = self.env.reset()
        self.rewards = {}
        self.transits = {}
        self.values = np.zeros(self.env.observation_space.n)

    def update_transits_rewards(self, state, action, new_state, reward):
        key = (state, action)
        if key not in self.rewards:
            self.rewards[key] = 0
        self.rewards[key] += reward
        if key not in self.transits:
            self.transits[key] = {}
        if new_state not in self.transits[key]:
            self.transits[key][new_state] = 0
        self.transits[key][new_state] += 1

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            self.env.render()
            new_state, reward, done, truncated, info = self.env.step(action)
            self.update_transits_rewards(self.state, action, new_state, reward)
            self.state = new_state
        self.env.close()

    def print_value_table(self):
       print(self.values)

    def extract_policy(self):
        policy = [np.zeros(self.env.observation_space.n, dtype=int)]
        for state in range(self.env.observation_space.n):
            policy[state] = self.select_action(state)
        return policy

    def calc_action_value(self, state, action):
        action_values = np.zeros(self.env.action_space.n)
        for action in range(self.env.action_space.n):
            for probability, next_state, reward, finished in self.env.P[state][action]:
                action_values[action] += probability * (reward + GAMMA * self.values[next_state])
        return np.argmax(action_values)

    def select_action(self, state):
        best_action = self.env.action_space.sample()
        best_action_value = -float("-inf")
        for action in range(self.env.action_space.ns):
            action_value = self.calc_action_value(state, action)
            if action_value > best_action_value:
                best_action_value = action_value
                best_action = action
        return best_action

    def play_episode(self):
        total_reward = 0
        state = self.env.reset()
        while not done:
            action = self.select_action(state)
            new_state, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            self.update_transits_rewards(state, action, new_state, reward)
            if done or truncated:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            best_action = self.env.action_space[0]
            best_action_value = self.calc_action_value(state, best_action)
            for action in range(1, self.env.action_space.n):
                action_value = self.calc_action_value(state, action)
                if action_value > best_action_value:
                    best_action_value = action_value
            self.values[state] = best_action_value

if __name__ == "__main__":
    agent = Agent()

    iter_no = 0
    best_reward = -float('inf')
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = sum(agent.play_episode(agent.env) for _ in range(TEST_EPISODES)) / TEST_EPISODES
        
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))

        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            agent.print_value_table()
            policy = agent.extract_policy()
            print(policy)
            break