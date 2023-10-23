# Malachi Eberly
# Assignment 4: Frozen Lake Value Iteration

import gymnasium as gym
import collections

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20
SEED = 42

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME, desc=None, map_name="4x4", render_mode = "human", is_slippery=True)
        self.state = self.env.reset(seed=SEED)
        self.rewards = {}
        self.transits = {}
        self.values = [0.0] * self.env.observation_space.n

    def update_transits_rewards(self, state, action, new_state, reward):
        key = (state, action)
        if type(state) == tuple:
            key = (state[0], action)
        if key not in self.transits:
            self.transits[key] = {}
        self.rewards[key, new_state] = reward
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
            if done:
                self.state = self.env.reset(seed=SEED)
        self.env.close()

    def print_value_table(self):
        print("Value Table:")
        for state in range(self.env.observation_space.n):
            if state % 4 == 0:  # Assuming the FrozenLake environment has a 4x4 grid
                print("\n")
            print(f"State {state}: {self.values[state]:.3f}\t", end="")
        print("\n")

    def extract_policy(self):
        policy = {}
        for state in range(self.env.observation_space.n):
            best_action = self.select_action(state)
            policy[state] = best_action
        return policy

    def calc_action_value(self, state, action):
        action_value = 0.0
        for new_state in range(self.env.observation_space.n):
            transition_prob = self.transits[(state, action)][new_state] / sum(self.transits[(state, action)].values())
            immediate_reward = self.rewards.get((state, action, new_state), 0.0)
            future_reward = self.values[new_state]
            action_value += transition_prob * (immediate_reward + GAMMA * future_reward)
        return action_value

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
        state = self.env.reset(seed=SEED)
        done = False
        while not done:
            action = self.select_action(state)
            new_state, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            self.update_transits_rewards(state, action, new_state, reward)
            if done or truncated:
                print("Ended")
                break
            state = new_state
        return total_reward


    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = []
            for action in range(self.env.action_space.n):
                action_value = self.calc_action_value(state, action)
                state_values.append(action_value)
            self.values[state] = max(state_values)
            
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