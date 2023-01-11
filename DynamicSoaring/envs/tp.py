# from DynamicSoaring.envs.env import Environment
import numpy as np
import time
from matplotlib import pyplot as plt
# from scipy.io import loadmat

# env = Environment()

# done = False
# x = []
# y = []
# z = []
# # while not done:
# #     state, reward, done, _ = env.step(np.random.random(3)*2 - 1)
# #     print(state, reward, done)
# #     x.append(state[0])
# #     y.append(state[1])
# #     z.append(state[2])

# mat = loadmat('/home/suneet/Desktop/RL/project/Environment-Dynamic-Soaring/DynamicSoaring/envs/Differential_flatness_states.mat')
# target_X = mat['X']
# v = []
# gamma = []
# chi = []

# for i in range(target_X.shape[0]):
#     x.append(target_X[i][3])
#     y.append(target_X[i][4])
#     z.append(target_X[i][5])
#     v.append(target_X[i][0])
#     gamma.append(target_X[i][1])
#     chi.append(target_X[i][2])

# print(max(x), min(x))
# print(max(y), min(y))
# print(max(z), min(z))
# print(max(v), min(v))
# print(max(chi), min(chi))
# print(max(gamma), min(gamma))

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(x, y, z, 'gray')

# # plt.plot((x,y))
# plt.show()

x=0
for i in range(40):

    x=np.sin(i)
    y = np.cos(30*x-5)

    plt.scatter(x, y)

    plt.title("Scatter Plot")

    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")

    plt.pause(0.2)

# Display

plt.show()
# Display

while not done:
    state, reward, done, _ = env.step(action)
    
    x.append(state[0])
    y.append(state[1])
    z.append(state[2])



# importing libraries
import numpy as np
import time
import matplotlib.pyplot as plt
 
# creating initial data values
# of x and y
x = []
y = []
 
# to run GUI event loop
plt.ion()
 
# here we are creating sub plots
figure, ax = plt.subplots(1,3)
line1, = ax.plot(x, y)


# setting title
plt.title("Geeks For Geeks", fontsize=20)
 
# setting x-axis label and y-axis label
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
 
# Loop
for _ in range(50):
    # creating new Y values
    new_y = np.sin(x-0.5*_)
 
    # updating data values
    line1.set_xdata(x)
    line1.set_ydata(new_y)
 
    # drawing updated values
    figure.canvas.draw()
 
    # This will run the GUI event
    # loop until all UI events
    # currently waiting have been processed
    figure.canvas.flush_events()
 
    time.sleep(0.1)