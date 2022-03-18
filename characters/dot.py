import pygame
import numpy as np

class Dot(pygame.sprite.Sprite):
    def __init__(self, game, row=1, col=1):
        super(Dot, self).__init__()
        self.game = game
        self.reward = 1
        self.size = 14
        self.row = row
        self.col = col

        self.image = pygame.image.load('assets/dot.png')
        self.image = pygame.transform.scale(self.image, (self.size, self.size))

        self.rect= self.image.get_rect()
        self.rect.x = 27*col - self.size//2 + 13
        self.rect.y = int(24.4*row) - self.size//2 + 13
        self.check()
    
    def check(self):
        if np.any(self.game.map[self.rect.y:self.rect.y+self.size, self.rect.x:self.rect.x+self.size]):
            self.game.all_dots.remove(self)
            self.game.np_all_dots[self.row-1, self.col-1] = 0
            