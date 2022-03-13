import pygame
import numpy as np
import matplotlib.pyplot as plt

class PacMan(pygame.sprite.Sprite):
    def __init__(self, game):
        super(PacMan, self).__init__()
        self.game = game

        self.score = 0

        self.velocity = 1
        self.size = 20

        self.image = pygame.image.load('assets/pacman.png')
        self.image = pygame.transform.scale(self.image, size = (self.size, self.size))

        self.images = {
            'up': pygame.transform.scale(pygame.image.load('assets/pacman/up.png'), size= (self.size, self.size)),
            'down': pygame.transform.scale(pygame.image.load('assets/pacman/down.png'), size= (self.size, self.size)),
            'left': pygame.transform.scale(pygame.image.load('assets/pacman/left.png'), size= (self.size, self.size)),
            'right': pygame.transform.scale(pygame.image.load('assets/pacman/right.png'), size= (self.size, self.size)),
        }

        self.rect = self.image.get_rect()

        self.rect.x = 512//2 - 10
        self.rect.y = 512 - 47

    def eat(self):
        for dot in self.game.check_collision(self, self.game.all_dots):
            self.game.all_dots.remove(dot)
            self.score += dot.reward
            print(self.score)

    def move_right(self):
        # print(self.image.get_rect())
        self.image = self.images['right']
        if not np.any(self.game.map[self.rect.y:self.rect.y+self.size, self.rect.x + self.size+self.velocity]):
            self.rect.x += self.velocity
            self.eat()

    def move_left(self):
        self.image = self.images['left']
        if not np.any(self.game.map[self.rect.y:self.rect.y+self.size, self.rect.x -self.velocity]):
            self.rect.x -= self.velocity
            self.eat()

    def move_up(self):
        self.image = self.images['up']
        if not np.any(self.game.map[self.rect.y-self.velocity, self.rect.x:self.rect.x+self.size]):
            self.rect.y -= self.velocity
            self.eat()

    def move_down(self):
        self.image = self.images['down']
        if not np.any(self.game.map[self.rect.y+self.size+self.velocity, self.rect.x:self.rect.x+self.size]):
            self.rect.y += self.velocity
            self.eat()

