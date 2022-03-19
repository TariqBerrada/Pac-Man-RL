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


class Environment():
    def __init__(self):
        
        self.scale = 15
        self.w, self.h = 21*self.scale, 19*self.scale

        self.map = plt.imread('assets/map.png')
        self.map = cv2.resize(self.map, (self.h, self.w))

        self.grid = joblib.load('assets/grid_sparse.pt')

        self.dot = plt.imread('assets/dot.png', format = 'PNG')
        self.dot = cv2.resize(self.dot, (self.scale, self.scale))
        self.cell_values = np.zeros_like(self.grid, dtype = int)
        
        self.populate_dots()

        self.pacman = Pacman(self)
        for i in range(20):
            self.pacman.move_right()
            self.pacman.eat()

            self.draw(self.pacman.image, self.pacman.y, self.pacman.x)
            # print(self.map)
            plt.figure()
            # plt.imshow(self.map)
            plt.imsave("anim/img_%.3d.jpg"%i, self.map.clip(0, 1))
            plt.close()



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

    

if __name__ == '__main__':
    env = Environment()