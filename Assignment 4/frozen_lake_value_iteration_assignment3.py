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
        self.num_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.n
        self.values = np.zeros(self.num_states)

    def update_transits_rewards(self, state, action, new_state, reward):
        # Get the key, which is a state action pair
        # update rewards which is accessed by key plus the new state
        # update transits count which is accessed by key and new_state
        pass

    def play_n_random_steps(self, count):
        for _ in range(count):
            # get an action
            # step through the environment
            # update the transits rewards
            # update the state
            pass

    def print_value_table(self):
       print(self.values)

    def extract_policy(self):
        policy = [np.zeros(self.num_states, dtype=int)]
        for state in range(self.num_states):
            policy[state] = self.select_action(state)
        return policy

    def calc_action_value(self, state, action):
        action_values = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            for probability, next_state, reward, finished in self.env.P[state][action]:
                action_values[action] += probability * (reward + GAMMA * self.values[next_state])
        return np.argmax(action_values)

    def select_action(self, state):
        best_action = 0
        best_action_value = self.calc_action_value(state, best_action)
        for action in range(1, self.num_actions):
            action_value = self.calc_action_value(state, action)
            if action_value > best_action_value:
                best_action_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        # define reward and state
        # While loop
            # select an action
            # take a step
            # if state is multiple
                # update reward
                # update count
            # else
                # update reward
                # update count
            # update total reward
            # get out if we're done
            # set state to new state
        # return total reward
        pass

    def value_iteration(self):
        action_values = []
        for state in self.num_states:
            self.select_action(state)
        self
        # for each state
            # set state_values equalt to a list of calc_action_value for every action
        # set self values to the max state_values         

        pass


if __name__ == "__main__":
    agent = Agent()

    iter_no = 0
    best_reward = None
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = None # sum of play episode for all 20 episodes / number of episodes
        
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))

        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            agent.print_value_table()
            policy = agent.extract_policy()
            print(policy)
            break
