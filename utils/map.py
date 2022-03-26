import numpy as np
import cv2, joblib
import matplotlib.pyplot as plt
from skimage.measure import block_reduce

_map = plt.imread('assets/map.png')

h, w = 21*9, 19*9

# map_r = cv2.cvtColor(cv2.resize(_map, (h, w)), cv2.COLOR_BGR2GRAY)
map_r = cv2.cvtColor(_map, cv2.COLOR_BGR2GRAY)
map_r[:, ::72] = 1
map_r[::72, :] = 1

# grid = np.zeros((h//9, w//9), dtype = bool)

grid = block_reduce(map_r, (72, 72), np.median)[:-1, :-1] > 0
joblib.dump(grid, "assets/grid_sparse.pt")

fig, ax = plt.subplots(1, 2)
ax[0].imshow(map_r)
ax[1].imshow(grid > .3)
plt.show()