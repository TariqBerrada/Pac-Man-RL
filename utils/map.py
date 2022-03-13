import numpy as np
import cv2, joblib
import matplotlib.pyplot as plt

_map = plt.imread('assets/map.png')
# print(_map)
# resolution = 50
# dx = _map.shape[1]//resolution
# dy =  _map.shape[0]//resolution
# print(dx, dy)
# iter_x = range(0, _map.shape[1], dx)
# iter_y = range(0, _map.shape[0], dy)

# grid = np.zeros((len(iter_y), len(iter_x)))

# for y, i in enumerate(iter_y):
#     for x, j in enumerate(iter_x):
#         # print(np.max(_map[i:i+dy, j:j+dx]))
#         if np.max(_map[i:i+dy, j:j+dx]) > 0:
#             grid[y, x] = 1

# fig, ax = plt.subplots(1, 2)

# ax[0].imshow(_map)
# ax[1].imshow(grid, cmap = 'gray')
# plt.show()

grid = cv2.cvtColor(cv2.resize(_map, (512, 512)), cv2.COLOR_RGB2GRAY)

joblib.dump(grid > 0, "assets/grid.pt")

fig, ax = plt.subplots(1, 2)
print(_map.shape)
ax[0].imshow(_map)
ax[1].imshow(grid > 0, cmap = 'gray')
plt.show()