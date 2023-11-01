# Malachi Eberly
# Assignment 5: Frozen Lake Q-Learning

#!/usr/bin/env python3
import gymnasium as gym
from collections import defaultdict
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME, desc=None, map_name="4x4", is_slippery=False)
        self.state = self.env.reset()
        self.q_table = defaultdict(float)
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n

    def sample_env(self):
        action = self.env.action_space.sample()
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        old_state = self.state
        if terminated or truncated:
            self.state = self.env.reset()
        return old_state, action, reward, new_state

    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in range(self.n_actions):
            state = state[0] if type(state) is tuple else state
            action_value = self.q_table[(state, action)]
            if best_value is None or action_value > best_value:
                best_value, best_action = action_value, action
        return best_value, best_action

    def value_update(self, state, action, reward, new_state):
        best_new_value, _ = self.best_value_and_action(new_state)
        new_q_value = reward + (GAMMA * best_new_value)
        state = state[0] if type(state) is tuple else state
        self.q_table[(state, action)] += ALPHA * (new_q_value - self.q_table[(state, action)])

    def play_episode(self):
        total_reward = 0.0
        self.state = self.env.reset()
        while True:
            _, action = self.best_value_and_action(self.state)
            new_state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
            self.state = new_state
        return total_reward

    def print_values(self):
        # # Print the Q-values in a readable format
        # Hint: You can use nested loops to iterate over states and actions
        pass

    def print_policy(self):
        policy = {}
        for state in range(self.n_states):
            _, best_action = self.best_value_and_action(state)
            policy[state] = best_action
        print("State:", state, "has best action:", best_action)
        return policy

if __name__ == "__main__":
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
            cumulative_reward += agent.play_episode()
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
