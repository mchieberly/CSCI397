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
            q_values = np.zeros(self.num_actions)
            for action in range(self.num_actions):
                continue
            # append action to the policy
        return policy

    def calc_action_value(self, state, action):
        # get target counts which access transits by state, action
        # get the sum of all the counts
        # for each target state
            # calculate the proportion of reward plus gamma * value of the target state, then sum it all together. 
        # return that sum
        pass

    def select_action(self, state):
        # define best action and best value
        # For action in the range of actions
            # calculate the action value
            # if best value is less than action value
                # update best value and best action
        # return best action
        pass

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
