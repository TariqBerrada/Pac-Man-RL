import pygame
import time
from game import Game

import matplotlib.pyplot as plt

pygame.init()

h, w = 512, 512

pygame.display.set_caption('Pac-Man')
screen = pygame.display.set_mode((w, h))

# set background.
background = pygame.image.load('assets/map.png')
background = pygame.transform.scale(background, (512, 512))

game = Game()

running = True
while True:
    screen.blit(background, dest = (0, 0))

    game.update(screen)
    ret = pygame.display.flip()
    # pygame.image.save(screen, 'temp.jpg')
    arr = pygame.surfarray.array3d(screen).transpose(1, 0, 2)
    # print(arr.shape)
    plt.imsave('temp.jpg', arr)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
        elif event.type == pygame.KEYDOWN:
            game.pressed[event.key] = True

            # if event.key == pygame.K_RIGHT:
            #     game.pacman.move_right()
            # if event.key == pygame.K_LEFT:
            #     game.pacman.move_left()
            # if event.key== pygame.K_UP:
            #     game.pacman.move_up()
            # if event.key == pygame.K_DOWN:
            #     game.pacman.move_down()
        elif event.type == pygame.KEYUP:
            game.pressed[event.key] = False
    # time.sleep(0.01)