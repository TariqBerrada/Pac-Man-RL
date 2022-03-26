import time
from game import Game
from environment_npy import PACMAN_Environment
import numpy as np
import matplotlib.pyplot as plt

N = 2

env = PACMAN_Environment()
env.env_init()
k=0
for i in range(N):

    print(f"Episode {i+1}/{N}")

    env.env_start()
    termination = False

    max_step = 1000
    j = 0
    while j < max_step and not termination:
        action = np.random.choice(["right", "left", "up", "down"])# env.next_action_keyboard()
        
        # Actions possibles : "right", "left", "up", "down", "quit"
        env.env_step(action)

        reward, state, termination = env.reward_state_term
        plt.imsave("anim/_img%.3d.jpg"%k, state.clip(0, 1))
        k += 1
        j += 1
    print("Score final : " + str(env.pacman.score))

    env.env_end(reward)
    env.env_cleanup()