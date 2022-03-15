import pygame
import time, cv2

import numpy as np

from game import Game
from environment import PACMAN_Environment
from agents.dqn import Agent

import matplotlib.pyplot as plt

num_episodes = 2
imsize = 224

# Initialize environment.
env = PACMAN_Environment()
env.env_init()

# Intialize agent.
agent = Agent(gamma=.95, epsilon = 1.0, batch_size= 512, n_actions =4, eps_dec=5e-5, eps_end=1e-4, input_dims= [224, 224], lr = 1e-3)

action_map = {
    0: "left",
    1: "right",
    2: "up",
    3: "down"
}

# For plots and figures.
scores, eps_history = [], []
avg_scores = []
len_history, avg_len_history = [], []

for i in range(num_episodes):

    print(f"Episode {i+1}/{num_episodes}")

    observation = env.env_start()
    observation = cv2.resize(observation, (imsize, imsize))
    termination = False
    score, ep_len = 0, 0
    # need an initial observation.
    # while not termination:
    for _ in range(500): # cste number of steps for now.
        action = agent.choose_action(observation)
        A = action_map[action] # int to str
        env.env_step(A)

        reward, observation_, termination = env.reward_state_term
        observation_ = cv2.resize(observation_, (imsize, imsize))
    
        agent.store_transition(observation, action, reward, observation_, termination)
        agent.learn()
        observation = observation_

        score += reward
        ep_len += 1

    print("Score final : " + env.pacman.score)

    env.env_end(reward)
    env.env_cleanup()

    eps_history.append(agent.epsilon)
    avg_score = np.mean(scores[-100:])
    avg_len = np.mean(len_history[-100:])

    len_history.append(ep_len)
    avg_len_history.append(avg_len)
    scores.append(score)
    avg_scores.append(avg_score)

    if i%500 == 0:
        print(f'ep {i} - score {score} - avg_score {avg_score} - episode {avg_len} - epsilon {agent.epsilon}')
        agent.save()
    if i%50 == 0:
        fig, ax = plt.subplots(1, 2, figsize = (15, 4))
        ax[0].plot(scores, label = 'scores')
        ax[0].plot(avg_scores, label = 'scores (average)')
        ax[0].set_xlabel('episode')
        ax[0].set_ylabel('score')
        ax[0].set_title('Deep Q-learning score evolution')
        ax[0].legend()


        ax[1].plot(len_history, label = "episode length")
        ax[1].plot(avg_len_history, label = "episode length (average")
        ax[1].set_xlabel('episode')
        ax[1].set_ylabel('length')
        ax[1].set_title('Deep Q-learning episode length evolution')
        ax[1].legend()
        # ax[1].semilogy(eps_history)
        # ax[1].set_xlabel('episode')
        # ax[1].set_ylabel('$\epsilon$')
        plt.savefig('figures/dqn.jpg')
        plt.close()