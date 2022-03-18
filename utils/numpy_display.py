import numpy as np
import matplotlib.pyplot as plt
import cv2

def display(env):
    plt.figure()
    background = env.np_background
    background = cv2.resize(background, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    for i in range(env.np_all_dots.shape[0]):
        for j in range(env.np_all_dots.shape[1]):
            if env.np_all_dots[i,j]:
                mask = plt.imread('assets/dot.png')
                mask = cv2.resize(mask, dsize=(14, 14), interpolation=cv2.INTER_CUBIC)
                # plt.imshow(mask, cmap='gray')
                background[27*i - 14//2 + 13:27*i - 14//2 + 13+mask.shape[0], int(24.4*j) - 14//2 + 13:int(24.4*j) - 14//2 + 13+mask.shape[1]] = mask[:,:,0:3]
                # plt.imshow(dot.rect, dot.rect.x, dot.rect.y)

    plt.imshow(background)
    plt.show()