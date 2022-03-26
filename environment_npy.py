import imp
import joblib
import cv2

import numpy as np

import matplotlib.pyplot as plt

from characters.pacman_npy import Pacman
from characters.ghost_npy import Ghost


class PACMAN_Environment():
    """Implements the environment for an RLGlue environment
    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    def __init__(self):
        reward = None
        observation = None
        termination = None
        self.reward_obs_term = (reward, observation, termination)

    def env_init(self):
        """Setup for the environment called when the experiment first starts.
        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """
        reward = None
        state = None
        termination = None
        self.reward_state_term = (reward, state, termination)
        self.scale = 15

        self.scale = 15
        self.w, self.h = 21*self.scale, 19*self.scale

        
    def draw(self, sprite, y, x):
        ly, lx = self.scale*y, self.scale*x
        p1 = np.einsum('ijk,ij->ijk', sprite[..., :3], sprite[..., -1])
        p2 = np.einsum('ijk,ij->ijk', self.map[ly:ly+self.scale, lx:lx+self.scale], sprite[..., -1])
        
        self.map[ly:ly+self.scale, lx:lx+self.scale] = p1 + p2

    def populate_dots(self):
        self.dots = np.zeros_like(self.grid, dtype = bool)
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if not self.grid[i, j]:
                    self.draw(self.dot, i, j)
                    self.cell_values[i, j] = 1

    def env_start(self, nrow=21, ncol = 19):
        """The first method called when the experiment starts, called before the
        agent starts.
        Returns:
            The first state observation from the environment.
        """
        reward = 0

        self.map = plt.imread('assets/map.png')
        self.map = cv2.resize(self.map, (self.h, self.w))

        self.grid = joblib.load('assets/grid_sparse.pt')
    
        self.dot = plt.imread('assets/dot.png', format = 'PNG')
        self.dot = cv2.resize(self.dot, (self.scale, self.scale))
        self.cell_values = np.zeros_like(self.grid, dtype = int)

        self.populate_dots()

        self.pacman = Pacman(self)

        self.ghosts = [Ghost(self, color = c) for c in ["red", "purple", "green"]]

        termination = False
        state = self.map
        self.reward_state_term = (reward, state, termination)

        # return initial state.
        return state

    
    def env_end(self, reward):
        """

        """
        pass

    def env_step(self, action):
        """A step taken by the environment.
        Args:
            action: The action taken by the agent
        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
        last_score = self.pacman.score

        if action == "right" and self.pacman.x + 1 < self.w//self.scale-1:
            self.pacman.move_right()
        if action == "left" and self.pacman.x > 0:
            self.pacman.move_left()
        if action == "up" and self.pacman.y > 0:
            self.pacman.move_up()
        if action == "down" and self.pacman.y + 1 < self.h//self.scale-1:
            self.pacman.move_down()

        win = False
        
        for ghost in self.ghosts:
            action_g = np.random.choice(["right", "left", "up", "down"])
            if action_g == "right" and ghost.x + 1 < self.w//self.scale-1:
                ghost.move_right()
            if action_g == "left" and ghost.x > 0:
                ghost.move_left()
            if action_g == "up" and ghost.y > 0:
                ghost.move_up()
            if action_g == "down" and ghost.y + 1 < self.h//self.scale-1:
                ghost.move_down()
            win += ghost.win

        
        reward = self.pacman.score - last_score
        termination = win
        state = self.map 

        self.reward_state_term = (reward, state, termination)
        return self.reward_state_term

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        reward = None
        state = None
        termination = None
        self.reward_state_term = (reward, state, termination)
        self.pacman.score = 0


    def env_message(self, message):
        """A message asking the environment for information
        Args:
            message: the message passed to the environment
        Returns:
            the response (or answer) to the message
        """
        pass