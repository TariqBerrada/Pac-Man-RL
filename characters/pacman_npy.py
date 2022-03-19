import numpy as np

import cv2, joblib

import matplotlib.pyplot as plt

class Pacman():
    def __init__(self, env):
        self.scale = 15
        self.score = 0
        self.velocity = 15

        self.env = env

        self.images = {
            'up': cv2.resize(plt.imread('assets/pacman/up.png'), (self.scale, self.scale)),
            'down': cv2.resize(plt.imread('assets/pacman/down.png'), (self.scale, self.scale)),
            'left': cv2.resize(plt.imread('assets/pacman/left.png'), (self.scale, self.scale)),
            'right': cv2.resize(plt.imread('assets/pacman/right.png'), (self.scale, self.scale))
        }

        self.image = self.images['up']

        self.x = (19//2)# *self.scale
        self.y = (21-2)# *self.scale
    
    def eat(self):
        self.score += self.env.cell_values[self.y, self.x]
        self.env.cell_values[self.y, self.x] = 0
        self.env.map[self.y*self.scale:(self.y+1)*self.scale, self.x*self.scale:(self.x+1)*self.scale] = 0.
    
    def move_right(self):
        # print(self.image.get_rect())
        self.image = self.images['right']
        self.env.map[self.y*self.scale:(self.y+1)*self.scale, self.x*self.scale:(self.x+1)*self.scale] = 0.
    
        # if not np.any(self.game.map[self.rect.y:self.rect.y+self.size, self.rect.x + self.size+self.velocity]):
        # x, y = self.x //self.velocity, self.rect.y//self.velocity
        x, y = self.x, self.y
        if x+1 < self.env.grid.shape[1] and not self.env.grid[y, x+1]:
            self.x += 1# self.velocity
            self.eat()
        self.env.draw(self.image, self.y, self.x)


    def move_left(self):
        self.image = self.images['left']
        self.env.map[self.y*self.scale:(self.y+1)*self.scale, self.x*self.scale:(self.x+1)*self.scale] = 0.
    
        # x, y = self.rect.x //self.velocity, self.rect.y//self.velocity
        x, y = self.x, self.y
        if x -1 > 0 and not self.env.grid[y, x-1]:
        # if not np.any(self.game.map[self.rect.y:self.rect.y+self.size, self.rect.x -self.velocity]):
            self.x -= 1 # self.velocity
            self.eat()
        self.env.draw(self.image, self.y, self.x)

    def move_up(self):
        self.image = self.images['up']
        self.env.map[self.y*self.scale:(self.y+1)*self.scale, self.x*self.scale:(self.x+1)*self.scale] = 0.
    
        # if not np.any(self.game.map[self.rect.y-self.velocity, self.rect.x:self.rect.x+self.size]):
        #x, y = self.rect.x //self.velocity, self.rect.y//self.velocity
        x, y = self.x, self.y
        if y-1 > 0 and not self.env.grid[y-1, x]:
            self.y -= 1 # self.velocity
            self.eat()
        self.env.draw(self.image, self.y, self.x)


    def move_down(self):
        self.image = self.images['down']
        self.env.map[self.y*self.scale:(self.y+1)*self.scale, self.x*self.scale:(self.x+1)*self.scale] = 0.
    
        # x, y = self.rect.x //self.velocity, self.rect.y//self.velocity
        x, y = self.x, self.y
        if y+1 < self.env.grid.shape[0] and not self.env.grid[y+1, x]:
        # if not np.any(self.game.map[self.rect.y+self.size+self.velocity, self.rect.x:self.rect.x+self.size]):
            self.y += 1 # self.velocity
            self.eat()
        self.env.draw(self.image, self.y, self.x)
