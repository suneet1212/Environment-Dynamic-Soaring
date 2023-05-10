from DynamicSoaring.envs.env import Environment
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from scipy.io import loadmat
# from scipy.interpolate import splrep, splev
import os
from scipy.interpolate import interp1d
import time
from mpl_toolkits.mplot3d import Axes3D
env1 = Environment()
env2 = Environment()

curr_dir = os.getcwd()
address = "models/ppo_shortened_60_02042023122445"
path = os.path.join(curr_dir, address)
mat = loadmat(os.path.join(curr_dir, 'DynamicSoaring/envs/Pseudospectral_states_2.mat'))
target_X = mat['X']
target_U = mat['U']  # [mu;CL;T]
target_tvec = mat['tvec']
mu = target_U[:,0]
cl = target_U[:,1]
T = target_U[:,2]

#   interpolate the actions and check how it is coming up:
target_t = target_tvec.reshape(-1)
# print(target_t)
start_t = target_t[0]
end_t = target_t[-1]

time_step = np.arange(start_t, end_t, step=0.01)
# print(target_t.shape)

f_mu = interp1d(target_t, mu, kind='cubic')
f_cl = interp1d(target_t, cl, kind='cubic')
f_T = interp1d(target_t, T, kind='cubic')

mu_fit = f_mu(time_step)
cl_fit = f_cl(time_step)
T_fit = f_T(time_step)

x1 = []
y1 = []
z1 = []

x2 = []
y2 = []
z2 = []
done = False
state = env1.reset()
_ = env2.reset()
plt.ion()
i = 0
tot_reward = 0
model = PPO.load(os.path.join(curr_dir, path, '5.zip'))

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# while not done:
#     # action = model.predict(state)
#     # print(action)
#     # print(action[0])
#     # print(action)
#     actual1 = env1.state_space_to_state()
#     actual_action = np.zeros(3)
#     actual_action[0] = env1.cl_to_act_space(cl_fit[i])
#     actual_action[1] = mu_fit[i]/env1.mu_scale
#     actual_action[2] = T_fit[i]/env1.thrust_scale
    
#     state1, reward1, done1, _ = env1.step(actual_action)
#     # print(reward, done)
#     # print(actual[0], actual[1], actual[2])
#     x1.append(actual1[0])
#     y1.append(actual1[1])
#     z1.append(actual1[2])
#     # tot_reward += reward

#     actual2 = env2.state_space_to_state()
#     action = np.zeros(3)
    
#     state, reward, done, _ = env2.step(action)
#     # print(reward, done)
#     # print(actual[0], actual[1], actual[2])
#     x2.append(actual2[0])
#     y2.append(actual2[1])
#     z2.append(actual2[2])
#     tot_reward += reward
    
#     print(x2[i],y2[i],z2[i])
#     # sc._offset3d = (x2,y2,z2)
#     ax.scatter(x1[i], y1[i], z1[i], c = 'r', marker='o')
#     ax.scatter(x2[i], y2[i], z2[i], c = 'b', marker='x')
#     i+=1
    
#     # fig.canvas.draw()
#     # fig.canvas.flush_events()
#     # plt.draw()
#     plt.pause(0.001)
#     # time.sleep(0.)

i = 0
_ = env1.reset()
x3 = []
y3 = []
z3 = []
tot_reward = 0
while not done:
    # action = model.predict(state)
    # print(action)
    # print(action[0])
    # print(action)
    actual = env1.state_space_to_state()
    action = np.zeros(3)
    action[0] = env1.cl_to_act_space(cl_fit[i])
    action[1] = mu_fit[i]/env1.mu_scale
    action[2] = T_fit[i]/env1.thrust_scale
    
    state, reward, done, _ = env1.step(action)
    # print(reward, done)
    # print(actual[0], actual[1], actual[2])
    x3.append(actual[0])
    y3.append(actual[1])
    z3.append(actual[2])
    tot_reward += reward
    i+=1

state = env1.reset()
x = []
y = []
z = []
i = 0
done = False
model = PPO.load(os.path.join(curr_dir, path, '1.zip'))
total_reward = 0
while not done:
    action = model.predict(state)
    # print(action[0])
    # print(action)
    actual = env1.state_space_to_state()
    state, reward, done, _ = env1.step(action[0])
    # print(reward, done)
    x.append(actual[0])
    y.append(actual[1])
    z.append(actual[2])
    i+=1
    total_reward += reward
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot3D(x3, y3, z3, label="actual")
ax.plot3D(x, y, z, label="5.zip")

# print(x)
plt.legend()
plt.pause(5)
plt.show()