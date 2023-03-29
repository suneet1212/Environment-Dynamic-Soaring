from DynamicSoaring.envs.env import Environment
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from scipy.io import loadmat
# from scipy.interpolate import splrep, splev
import os
from scipy.interpolate import interp1d
env = Environment()

address = "models/ppo_shortened_40_18032023105434"
curr_dir = os.getcwd()
path = os.path.join(curr_dir, address)
models = os.listdir(path)
# print(models)

models.remove('final.zip')
models = sorted(models, key=lambda x: int(os.path.splitext(x)[0]))
models.append('final.zip')
# print(models)

curr_dir = os.getcwd()
mat = loadmat(os.path.join(curr_dir, 'DynamicSoaring/envs/Pseudospectral_states_2.mat'))
target_X = mat['X']
target_U = mat['U']  # [mu;CL;T]
target_tvec = mat['tvec']
mu = target_U[:,0]
cl = target_U[:,1]
T = target_U[:,2]

## plot the trajectory directly
# state = env.reset()
# actual_state = env.state_space_to_state()
# print('actual')
# print(target_X[0])

x1 = []
y1 = []
z1 = []
for i in range(target_X.shape[0]):
    x1.append(target_X[i][3])
    y1.append(target_X[i][4])
    z1.append(target_X[i][5])
    # v.append(target_X[i][0])
    # ppo_rewardIsDistance_fixedStartgamma.append(target_X[i][1])
    # chi.append(target_X[i][2])

def plot_fig(a,b,i):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    for addr in models:
        if (addr.split('.'))[0] != 'final':
            num = int((addr.split('.'))[0])
            if (num >= a or num <= b):
                continue
        else:
            continue
        model = PPO.load(os.path.join(curr_dir, path, addr))
        str_list = addr.split('.')
        state = env.reset()
        done = False
        x = []
        y = []
        z = []
        i = 0
        tot_reward = 0
        while not done:
            action = model.predict(state)
            # print(action[0])
            # print(action)
            actual = env.state_space_to_state()
            state, reward, done, _ = env.step(action[0])
            # print(reward, done)
            x.append(actual[0])
            y.append(actual[1])
            z.append(actual[2])
            i+=1
            if i > 200:
                i = i
            if i > 300:
                i = i
            tot_reward += reward
        # print(i)
        # global ax
        ax.plot3D(x, y, z, label=str_list[0])
        print(addr, " ", tot_reward)

    state = env.reset()
    actual_state = env.state_space_to_state()
    # print('actual')
    global target_X
    # print(target_X[0])
    x1 = []
    y1 = []
    z1 = []
    for i in range(target_X.shape[0]):
        x1.append(target_X[i][3])
        y1.append(target_X[i][4])
        z1.append(target_X[i][5])
        # v.append(target_X[i][0])
        # ppo_rewardIsDistance_fixedStartgamma.append(target_X[i][1])
        # chi.append(target_X[i][2])
    ax.plot3D(x1, y1, z1, label='actual')
    # print(i)
# plt.legend()
# plt.show()

# Play out one model:
state = env.reset()
x = []
y = []
z = []
i = 0
done = False
model = PPO.load(os.path.join(curr_dir, path, '250.zip'))
total_reward = 0
while not done:
    action = model.predict(state)
    # print(action[0])
    # print(action)
    actual = env.state_space_to_state()
    state, reward, done, _ = env.step(action[0])
    # print(reward, done)
    x.append(actual[0])
    y.append(actual[1])
    z.append(actual[2])
    i+=1
    total_reward += reward


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

x2 = []
y2 = []
z2 = []
done = False
state = env.reset()

i = 0
tot_reward = 0
while not done:
    # action = model.predict(state)
    # print(action)
    # print(action[0])
    # print(action)
    actual = env.state_space_to_state()
    action = np.zeros(3)
    action[0] = env.cl_to_act_space(cl_fit[i])
    action[1] = mu_fit[i]/env.mu_scale
    action[2] = T_fit[i]/env.thrust_scale
    
    state, reward, done, _ = env.step(action)
    # print(reward, done)
    # print(actual[0], actual[1], actual[2])
    x2.append(actual[0])
    y2.append(actual[1])
    z2.append(actual[2])
    tot_reward += reward
    i+=1

print("actual action", tot_reward)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot3D(x, y, z, label="250.zip")
ax.plot3D(x1, y1, z1, label='actual_trajectory')
ax.plot3D(x2, y2, z2, label='take_actual_action')
plt.legend()
# v = []
# gamma = []
# chi = []

plot_fig(6,0,1)
plt.legend()
plot_fig(11,5,2)
plt.legend()
plot_fig(16,10,3)
plt.legend()
plot_fig(21,15,4)
plt.legend()
plt.show()

# model1 = PPO.load("./models/ppo_new/124.zip")
# done = False
# x = []
# y = []
# z = []

# state = env.reset()
# i = 0
# # mat = loadmat('/home/suneet/Desktop/RL/project/Environment-Dynamic-Soaring/DynamicSoaring/envs/Differential_flatness_states.mat')
# # target_X = mat['X']
# done = False
# while not done:
#     action = model1.predict(state)
#     # print(action[0])
#     # print(action)
#     state, reward, done, _ = env.step(action[0])
#     print(reward, done)
#     actual = env.state_space_to_state()
#     x.append(actual[0])
#     y.append(actual[1])
#     z.append(actual[2])
#     i+=1
# print(i)
# ax.plot3D(x1, y1, z1, 'blue')
# ax.plot3D(x, y, z, 'green')


