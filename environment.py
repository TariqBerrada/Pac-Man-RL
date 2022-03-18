import joblib
import pygame
import matplotlib.pyplot as plt
import numpy as np

from characters.pacman import PacMan
from characters.dot import Dot
from game import Game

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

        self.map = joblib.load('assets/grid.pt')

            # set background.
        self.background = pygame.image.load('assets/map.png')
        self.background = pygame.transform.scale(self.background, (512, 512))

        self.np_background = plt.imread('assets/map.png')
    


    def env_start(self, nrow=19, ncol = 17):
        """The first method called when the experiment starts, called before the
        agent starts.
        Returns:
            The first state observation from the environment.
        """
        reward = 0

        pygame.init()

        h, w = 512, 512

        pygame.display.set_caption('Pac-Man')
        self.screen = pygame.display.set_mode((w, h))


        # self.game = Game()

        # self.pacman = PacMan(self.game)
        self.pacman = PacMan(self)
        self.all_players = pygame.sprite.Group()
        self.all_players.add(self.pacman)

        self.all_dots = pygame.sprite.Group()
        self.np_all_dots = np.ones((nrow, ncol))
        for i in range(1,nrow+1):
            for j in range(1, ncol+1):
                self.all_dots.add(Dot(self, i, j))
        
        for dot in self.all_dots:
            dot.check()

        self.pressed = {}

        self.h = 512
        self.w = 512

        termination = False
        state = self.pacman.rect
        self.reward_state_term = (reward, state, termination)

        # return initial state.
        arr = pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
        return arr

    
    def env_end(self, reward):
        """

        """
        pygame.quit()

    def env_step(self, action):
        """A step taken by the environment.
        Args:
            action: The action taken by the agent
        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        self.screen.blit(self.background, dest = (0, 0))
        self.screen.blit(self.pacman.image, self.pacman.rect)

        self.all_dots.draw(self.screen)

        ret = pygame.display.flip()
        # pygame.image.save(screen, 'temp.jpg')
        arr = pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
        # print(arr.shape)
        plt.imsave('temp.jpg', arr)

        last_score = self.pacman.score

        self.all_dots.draw(self.screen)

        if action == "right" and self.pacman.rect.x + self.pacman.rect.width < self.w-21:
            self.pacman.move_right()
        if action == "left" and self.pacman.rect.x > 21:
            self.pacman.move_left()
        if action == "up" and self.pacman.rect.y > 21:
            self.pacman.move_up()
        if action == "down" and self.pacman.rect.y + self.pacman.rect.height < self.h-21:
            self.pacman.move_down()


        # reward = self.pacman.score
        reward = self.pacman.score - last_score
        termination = action == "quit"
        state = arr # self.pacman.rect ## stat is the image screen.

        self.reward_state_term = (reward, state, termination)

    def next_action_keyboard(self):

        action = ""

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                action = "quit"
            elif event.type == pygame.KEYDOWN:
                self.pressed[event.key] = True
            elif event.type == pygame.KEYUP:
                self.pressed[event.key] = False

        if self.pressed.get(pygame.K_RIGHT) and self.pacman.rect.x + self.pacman.rect.width < self.w-21:
            action = "right"
        if self.pressed.get(pygame.K_LEFT) and self.pacman.rect.x > 21:
            action = "left"
        if self.pressed.get(pygame.K_UP) and self.pacman.rect.y > 21:
            action = "up"
        if self.pressed.get(pygame.K_DOWN) and self.pacman.rect.y + self.pacman.rect.height < self.h-21:
            action = "down"

        return action

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        reward = None
        state = None
        termination = None
        self.reward_state_term = (reward, state, termination)


    def env_message(self, message):
        """A message asking the environment for information
        Args:
            message: the message passed to the environment
        Returns:
            the response (or answer) to the message
        """
    def check_collision(self, sprite, group):
        return pygame.sprite.spritecollide(sprite, group, False, pygame.sprite.collide_mask)