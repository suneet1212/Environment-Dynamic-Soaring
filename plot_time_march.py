import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d

mat = loadmat('/home/suneet/Desktop/RL/project/Environment-Dynamic-Soaring/DynamicSoaring/envs/time_march.mat')
x = mat['x_tmarch']
y = mat['y_tmarch']
z = mat['z_tmarch']
# for i in range(len(x)):
#     print(x[i], y[i], z[i])
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x, y, z, 'red')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()

