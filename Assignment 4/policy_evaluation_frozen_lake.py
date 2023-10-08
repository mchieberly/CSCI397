import numpy as np
import gymnasium as gym


def policy_evaluation(policy, env, discount_factor=1.0, theta=1e-5):
    """
    Evaluate a policy given an environment.
    """
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V


def policy_improvement(env, discount_factor=1.0):
    """
    Policy Improvement function.
    """
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    while True:
        V = policy_evaluation(policy, env, discount_factor)
        policy_stable = True
        for s in range(env.observation_space.n):
            chosen_a = np.argmax(policy[s])
            action_values = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (
                        reward + discount_factor * V[next_state]
                    )
            best_a = np.argmax(action_values)
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.action_space.n)[best_a]
        if policy_stable:
            return policy, V


# Create FrozenLake environment
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True)

# Apply policy iteration
policy, V = policy_improvement(env)

print("Policy:")
print("0: Move left\n1: Move down\n2: Move right\n3: Move up")
print(policy)
print("Value Function:")
print(V)
