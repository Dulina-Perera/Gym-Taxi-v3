import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from IPython.display import clear_output
from mpl_toolkits import mplot3d

num_points = 50
iter_cnt = 0

list_learning_rates = np.linspace(0, 1, num_points)
list_discount_factors = np.linspace(0, 1, num_points)


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Episode: {frame['episode']}")
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        time.sleep(0.001)


def rl_agent(learning_rate, discount_factor):
    global iter_cnt
    iter_cnt += 1
    print(iter_cnt)

    env = gym.make("Taxi-v3")

    # Initialize the Q-table with zeros
    state_size = env.observation_space.n
    action_size = env.action_space.n
    # print(f'State count: {state_size}\nAction count: {action_size}')

    q_table = np.zeros([state_size, action_size])
    epsilon = 1.0
    decay_rate = 0.005

    # Training variables
    num_episodes = 2000
    max_steps = 99  # Per episode

    # print("AGENT IS TRAINING...")

    for episode in range(num_episodes):
        # Reset the environment.
        state = env.reset()
        step = 0
        done = False

        for step in range(max_steps):
            # Exploration-exploitation trade-off
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore.
            else:
                action = np.argmax(q_table[state, :])  # Exploit.

            # Take an action and observe the reward.
            next_state, reward, done, info = env.step(action)

            # Q-learning algorithm
            q_table[state, action] += learning_rate * (
                        reward + discount_factor * np.max(q_table[next_state]) - q_table[state, action])

            # Update to our next state
            state = next_state

            # if done, finish episode
            if done:
                break

        # Decrease epsilon
        epsilon = np.exp(-decay_rate * episode)

    clear_output()
    # print(f"Q-table: {q_table}")
    # print(f"Training completed over {num_episodes} episodes.\n")
    time.sleep(1)
    clear_output()

    # Watch our trained agent.
    total_rewards, total_epochs, total_penalties = 0, 0, 0
    num_episodes = 100
    frames = []

    for episode in range(num_episodes + 1):
        state = env.reset()
        epochs, penalties, episode_reward = 0, 0, 0

        done = False

        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)

            episode_reward += reward

            if reward == -10:
                penalties += 1

            # Put each rendered frame into dict for animation
            frames.append({
                'frame': env.render(mode='ansi'),
                'episode': episode,
                'state': state,
                'action': action,
                'reward': reward
            })
            epochs += 1

        total_rewards += episode_reward
        total_penalties += penalties
        total_epochs += epochs

    return total_rewards / num_episodes

    # print(f"Results after {num_episodes} episodes:")
    # print(f"Average rewards per episode: {total_rewards / num_episodes}")
    # print(f"Average timesteps per episode: {total_epochs / num_episodes}")
    # print(f"Average penalties per episode: {total_penalties / num_episodes}")

    # print_frames(frames)


LR, DF = np.meshgrid(list_learning_rates, list_discount_factors)

# Compute rewards for each combination.
rewards_surface = np.zeros((num_points, num_points))

for i in range(num_points):
    for j in range(num_points):
        lr = LR[i, j]
        df = DF[i, j]
        rewards_surface[i][j] = rl_agent(lr, df)

# Create a 3D plot
ax = plt.axes(projection='3d')
ax.plot_surface(LR, DF, rewards_surface, cmap='plasma')

# Set labels and title
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Discount Factor')
ax.set_zlabel('Average Rewards per Episode')
ax.set_title('3D Surface Plot of Average Rewards per Episode')

plt.show()
