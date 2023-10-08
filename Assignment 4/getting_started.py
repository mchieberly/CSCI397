"""
In a way, Reinforcement Learning is the science of making optimal decisions using experiences.
Breaking it down, the process of Reinforcement Learning involves these simple steps: Observation of the environment
Deciding how to act using some strategy
Acting accordingly
Receiving a reward or penalty
Learning from the experiences and refining our strategy
Iterate until an optimal strategy is found
"""

import gymnasium as gym
from IPython.display import clear_output
from time import sleep

env = gym.make("Taxi-v3", render_mode="human")
env.reset()
env.render()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

state = env.encode(
    3, 1, 2, 0
)  # (taxi row, taxi column, passenger index, destination index)
print("State:", state)

env.s = state
env.render()

env.P[328]


epochs = 0
penalties, reward = 0, 0

frames = []  # for animation

done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, truncated, info = env.step(action)

    if reward == -10:
        penalties += 1

    # Put each rendered frame into dict for animation
    frames.append(
        {
            "frame": env.render(),
            "state": state,
            "action": action,
            "reward": reward,
        }
    )

    epochs += 1


print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        # print(frame["frame"])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(0.1)


print_frames(frames)
