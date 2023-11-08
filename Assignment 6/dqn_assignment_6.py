# Malachi Eberly
# Assignment 6: Implement Deep Q-Learning

import gymnasium as gym
import math
import os
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as torch_functional

# Leave this alone, solves a macOS issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Create the environment with Gym
env = gym.make("CartPole-v1")

# Set up matplotlib for dynamic plots
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# Determine whether to use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a named tuple to hold experience tuples
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory(object):
    # Initialize the replay memory with a fixed capacity
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    # Function to save a transition to memory
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    # Function to sample a batch of transitions from memory
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # Function to get the current size of the memory
    def __len__(self):
        return len(self.memory)

# Define the neural network architecture for DQN
class DQN(nn.Module):
    # Initialize DQN layers
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.hl1 = nn.Linear(inputs, 128)
        self.hl2 = nn.Linear(128, 256)
        self.hl3 = nn.Linear(256, outputs)

    # Define forward pass
    def forward(self, x):
        x = torch_functional.leaky_relu(self.hl1(x))
        x = torch_functional.leaky_relu(self.hl2(x))
        return self.hl3(x)

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 20
TAU = 0.001
LR = 0.0001
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# Initialize DQN policy and target networks
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

# Define epsilon-greedy action selection function
def select_action(state):
    global steps_done
    sample = random.random()

    # Calculate the epsilon threshold for exploring
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
    
    if sample > eps_threshold:
        # Exploitation
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # Exploration
        return torch.tensor([[random.randrange(n_actions)]], device = device, dtype = torch.long)
    
episode_durations = []

# Define function to plot episode durations
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    # Check memory capacity
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Prepare batch data by removing any None values
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute the q-values and expected q-values
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute the Huber loss
    loss = torch_functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the network and clip the gradient
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def main():
    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 500

    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

    print("Complete")
    print("Average Duration:", sum(episode_durations) / len(episode_durations))
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()