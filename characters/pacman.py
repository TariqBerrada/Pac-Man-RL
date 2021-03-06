import pygame
import numpy as np
import matplotlib.pyplot as plt

class PacMan(pygame.sprite.Sprite):
    def __init__(self, game):
        super(PacMan, self).__init__()
        self.game = game

        self.score = 0

        self.velocity = 15
        self.size = 15

        self.image = pygame.image.load('assets/pacman.png')
        self.image = pygame.transform.scale(self.image, size = (self.size, self.size))

        self.images = {
            'up': pygame.transform.scale(pygame.image.load('assets/pacman/up.png'), size= (self.size, self.size)),
            'down': pygame.transform.scale(pygame.image.load('assets/pacman/down.png'), size= (self.size, self.size)),
            'left': pygame.transform.scale(pygame.image.load('assets/pacman/left.png'), size= (self.size, self.size)),
            'right': pygame.transform.scale(pygame.image.load('assets/pacman/right.png'), size= (self.size, self.size)),
        }

        self.rect = self.image.get_rect()

        self.rect.x = (19//2)*self.size
        self.rect.y = (21-2)*self.size 

    def eat(self):
        for dot in self.game.check_collision(self, self.game.all_dots):
            self.game.all_dots.remove(dot)
            self.score += dot.reward

    def move_right(self):
        self.image = self.images['right']
        x, y = self.rect.x //self.velocity, self.rect.y//self.velocity
        if x+1 < self.game.map.shape[1] and not self.game.map[y, x+1]:

            self.rect.x += self.velocity
            self.eat()

    def move_left(self):
        self.image = self.images['left']
        x, y = self.rect.x //self.velocity, self.rect.y//self.velocity
        if x -1 > 0 and not self.game.map[y, x-1]:
            self.rect.x -= self.velocity
            self.eat()

    def move_up(self):
        self.image = self.images['up']
        x, y = self.rect.x //self.velocity, self.rect.y//self.velocity
        if y-1 > 0 and not self.game.map[y-1, x]:
            self.rect.y -= self.velocity
            self.eat()

    def move_down(self):
        self.image = self.images['down']
        x, y = self.rect.x //self.velocity, self.rect.y//self.velocity
        if y+1 < self.game.map.shape[0] and not self.game.map[y+1, x]:
            self.rect.y += self.velocity
            self.eat()

