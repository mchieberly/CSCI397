import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Imports
import random, datetime
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame

# Initialize environment
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

# Only need to move right and jump while moving right for optimality
env = JoypadSpace(
    env,
    [['right'],
    ['right', 'A']]
)

# Put env in manipulatable format
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)

# Reset
env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = None 
# Initialize Mario agent
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

# Number of episodes
episodes = 40000

for e in range(episodes):

    # Reset the state
    state = env.reset()

    while True:
        # Choose an action based on the state
        action = mario.act(state)

        # Step through the environment
        next_state, reward, done, info = env.step(action)

        # Save the results of that step in memory buffer
        mario.cache(state, next_state, action, reward, done)

        # Learn from memory buffer
        q, loss = mario.learn()

        # Log the step
        logger.log_step(reward, loss, q)

        # Move to the next state
        state = next_state

        # Break if died, ran out of time, etc. or completed the level
        if done or info['flag_get']:
            break

    # Log the episode
    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )
