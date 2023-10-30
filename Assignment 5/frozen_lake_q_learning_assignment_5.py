# Malachi Eberly
# Assignment 5: Frozen Lake Q-Learning

#!/usr/bin/env python3
import gymnasium as gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME, desc=None, map_name="4x4", is_slippery=True)
        self.state = self.env.reset()
        self.qtable = {}
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n

    def sample_env(self):
        action = self.env.action_space.sample()
        new_state, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.state = self.env.reset()
        return (self.state, action, reward, new_state)

    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in range(self.n_actions):

        # Calculate the Q-value for each state-action pair
        # Update best_value and best_action based on the calculated Q-value
        return best_value, best_action

    def value_update(self, state, action, reward, new_state):
        # Call the best_value_and_action function to get the best Q-value for the new state
        # Calculate the new Q-value using the reward, gamma, and best Q-value of the new state
        # Update the Q-value of the current state-action pair using alpha and the new Q-value
        pass

    def play_episode(self):
        total_reward = 0.0
        self.state = self.env.reset()
        terminated, truncated = False, False
        while True:
            best_value, best_action = self.best_value_and_action(self.state)
            new_state, reward, terminated, truncated, info = self.env.step(best_action)
            total_reward += reward
            if terminated or truncated:
                break
            state = new_state
        return total_reward

    def print_values(self):
        # Print the Q-values in a readable format
        # Hint: You can use nested loops to iterate over states and actions
        pass

    def print_policy(self):
        # Print the policy derived from the Q-values
        # Initialize an empty dictionary named policy
        # Iterate over all possible states in the environment
        # Call the best_value_and_action function to get the best action for each state
        # Update the policy dictionary with the state-action pair
        # Print the state and corresponding best action
        # Return the policy dictionary
        pass


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
