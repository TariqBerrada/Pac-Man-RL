import pygame, joblib

from characters.pacman import PacMan
from characters.dot import Dot

class Game():
    def __init__(self, nrow=21, ncol = 19):
        self.map = joblib.load('assets/grid_sparse.pt')
        self.all_players = pygame.sprite.Group()
        self.pacman = PacMan(self)
        self.all_players.add(self.pacman)

        self.all_dots = pygame.sprite.Group()
        for i in range(nrow):
            for j in range(ncol):
                self.all_dots.add(Dot(self, i, j))
        
        for dot in self.all_dots:
            dot.check()

        self.pressed = {}
        self.scale = 15

        self.h = 21*self.scale # 512
        self.w = 19*self.scale # 512


    def check_collision(self, sprite, group):
        return pygame.sprite.spritecollide(sprite, group, False, pygame.sprite.collide_mask)
    
    
    def update(self, screen, verbose):

        # add dots.
        if verbose > 1:
            self.all_dots.draw(screen)

        if self.pressed.get(pygame.K_RIGHT) and self.pacman.rect.x + self.pacman.rect.width < self.w-21:
            self.pacman.move_right()
        if self.pressed.get(pygame.K_LEFT) and self.pacman.rect.x > 21:
            self.pacman.move_left()
        if self.pressed.get(pygame.K_UP) and self.pacman.rect.y > 21:
            self.pacman.move_up()
        if self.pressed.get(pygame.K_DOWN) and self.pacman.rect.y + self.pacman.rect.height < self.h-21:
            self.pacman.move_down()