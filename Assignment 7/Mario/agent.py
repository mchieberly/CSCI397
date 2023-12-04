import torch
import random, numpy as np
from pathlib import Path

from neural import MarioNet
from collections import deque

"""
This is the agent class; it has two neural networks, online and target
"""
class Mario:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):

        # Initialize hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.curr_step = 0
        self.burnin = 1e5  
        self.learn_every = 3   
        self.sync_every = 1e4   

        self.save_every = 5e5   
        self.save_dir = save_dir

        # Use the GPU if available
        self.use_cuda = torch.cuda.is_available()

        # Initialize the neural networks
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')
        if checkpoint:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    """
    Use an epsilon-greedy policy to choose an action
    """
    def act(self, state):

        # exploration rate / 1 percentage for choosing a random action
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # (1 - exploration rate) / 1 percentage for choosing the action with the highest Q-value
        else:
            state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
            state = state.unsqueeze(0)
            # Use the online neural net for choosing an action
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()

        # deincrement the exploration rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx

    """
    Save experience in the memory buffer
    """
    def cache(self, state, next_state, action, reward, done):

        # Convert data to correct types
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
        reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])
        done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor([done])

        # Add the data as a tuple to the memory buffer
        self.memory.append( (state, next_state, action, reward, done,) )

    """
    Access a batch of memory for learning
    """
    def recall(self):

        # Randomly select the batch and return it in the correct format
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


    """
    Estimate the current Q value
    """
    def td_estimate(self, state, action):
        # Uses the online neural network
        current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action] # Q_online(s,a)
        return current_Q


    """
    Calculate the TD target
    """
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        # Use the online neural network to find the next state's Q value
        next_state_Q = self.net(next_state, model='online')
        # Choose the best action for that state
        best_action = torch.argmax(next_state_Q, axis=1)
        # Find the next state's Q value using the target neural network and the same best action
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        # Return R + (gamma * next_Q) if not done, 0 otherwise
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()


    """
    Update the online network
    """
    def update_Q_online(self, td_estimate, td_target) :
        # Calculate the loss and optimize
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    """
    Set the weights of the target network to those of the online network
    """
    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    """
    Learn performance of the network and update the online network
    accordingly, using other methods
    """
    def learn(self):
        # Sync the network weights
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        # Save the model
        if self.curr_step % self.save_every == 0:
            self.save()

        # Return Nones depending on current step
        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Recall from memory buffer
        state, next_state, action, reward, done = self.recall()

        # Estimate the current Q-value
        td_est = self.td_estimate(state, action)

        # Calculate the target
        td_tgt = self.td_target(reward, next_state, done)

        # Calculate the loss and update the online network
        loss = self.update_Q_online(td_est, td_tgt)

        # return the Q-value and loss
        return (td_est.mean().item(), loss)

    """
    Save the model
    """
    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")


    """
    Load a model
    """
    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
