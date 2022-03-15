import pygame
import time
from game import Game
from environment import PACMAN_Environment

import matplotlib.pyplot as plt

N = 2

env = PACMAN_Environment()
env.env_init()

for i in range(N):

    print(f"Episode {i+1}/{N}")

    env.env_start()
    termination = False

    while not termination:
        # Draw screen

        action = "left" # env.next_action_keyboard()
        
        # Actions possibles : "right", "left", "up", "down", "quit"
        env.env_step(action)

        reward, state, termination = env.reward_state_term
        print(state.shape)

    print("Score final : " + env.pacman.score)

    env.env_end(reward)
    env.env_cleanup()

   # time.sleep(0.01)