import numpy as np

import cv2, joblib

import matplotlib.pyplot as plt

class Ghost():
    def __init__(self, env, color = 'red', damage = 100):
        self.scale = 15
        self.score = 0
        self.velocity = 15
        self.damage = damage

        self.env = env

        self.images = {
            'up': cv2.resize(plt.imread('assets/ghost_%s.png'%color), (self.scale, self.scale)),
            'down': cv2.resize(plt.imread('assets/ghost_%s.png'%color), (self.scale, self.scale)),
            'left': cv2.resize(plt.imread('assets/ghost_%s.png'%color), (self.scale, self.scale)),
            'right': cv2.resize(plt.imread('assets/ghost_%s.png'%color), (self.scale, self.scale))
        }

        self.image = self.images['up']

        self.x = (19//2)
        self.y = (21//2)

        self.win = False
    
    def eat(self):
        self.score += self.env.cell_values[self.y, self.x]
        self.env.cell_values[self.y, self.x] = 0
        self.env.map[self.y*self.scale:(self.y+1)*self.scale, self.x*self.scale:(self.x+1)*self.scale] = 0.
    
    def attack(self):
        if self.x == self.env.pacman.x and self.y == self.env.pacman.y:
            self.env.pacman.score -= self.damage
            self.win = True
    
    def move_right(self):
        self.image = self.images['right']
        self.env.map[self.y*self.scale:(self.y+1)*self.scale, self.x*self.scale:(self.x+1)*self.scale] = 0.
        if self.env.cell_values[self.y, self.x] > 0:
            self.env.draw(self.env.dot, self.y, self.x)
        
        x, y = self.x, self.y
        if x+1 < self.env.grid.shape[1] and not self.env.grid[y, x+1]:
            self.x += 1
        self.env.draw(self.image, self.y, self.x)
        self.attack()

    def move_left(self):
        self.image = self.images['left']
        self.env.map[self.y*self.scale:(self.y+1)*self.scale, self.x*self.scale:(self.x+1)*self.scale] = 0.
        if self.env.cell_values[self.y, self.x] > 0:
            self.env.draw(self.env.dot, self.y, self.x)
        
        x, y = self.x, self.y
        if x -1 > 0 and not self.env.grid[y, x-1]:
            self.x -= 1 
        self.env.draw(self.image, self.y, self.x)
        self.attack()

    def move_up(self):
        self.image = self.images['up']
        self.env.map[self.y*self.scale:(self.y+1)*self.scale, self.x*self.scale:(self.x+1)*self.scale] = 0.
        if self.env.cell_values[self.y, self.x] > 0:
            self.env.draw(self.env.dot, self.y, self.x)
        
        x, y = self.x, self.y
        if y-1 > 0 and not self.env.grid[y-1, x]:
            self.y -= 1 
        self.env.draw(self.image, self.y, self.x)
        self.attack()

    def move_down(self):
        self.image = self.images['down']
        self.env.map[self.y*self.scale:(self.y+1)*self.scale, self.x*self.scale:(self.x+1)*self.scale] = 0.
        if self.env.cell_values[self.y, self.x] > 0:
            self.env.draw(self.env.dot, self.y, self.x)
    
        x, y = self.x, self.y
        if y+1 < self.env.grid.shape[0] and not self.env.grid[y+1, x]:
            self.y += 1 
        self.env.draw(self.image, self.y, self.x)
        self.attack()