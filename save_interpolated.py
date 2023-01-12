from scipy.io import loadmat, savemat
from scipy.interpolate import interp1d
import numpy as np
from DynamicSoaring.envs.env import Environment
from matplotlib import pyplot as plt

mat = loadmat('/home/suneet/Desktop/RL/project/Environment-Dynamic-Soaring/DynamicSoaring/envs/Pseudospectral_states_2.mat')
target_X = mat['X']
target_U = mat['U']  # [mu;CL;T]
target_tvec = mat['tvec']
mu = target_U[:,0]
cl = target_U[:,1]
T = target_U[:,2]

target_t = target_tvec.reshape(-1)
start_t = target_t[0]
end_t = target_t[-1]

target_t = target_tvec.reshape(-1)
f_mu = interp1d(target_t, mu, kind='cubic')
f_cl = interp1d(target_t, cl, kind='cubic')
f_T = interp1d(target_t, T, kind='cubic')

time_step = np.arange(start_t, end_t, step=0.01)

mu_fit = f_mu(time_step)
cl_fit = f_cl(time_step)
T_fit = f_T(time_step)

# print(mu_fit.shape)
interpolated_U = np.stack((mu_fit, cl_fit, T_fit), axis=1)
# print(a)
# print(mu_fit)

### To save the interpolated values to a matrix
# u_dict = {'u': interpolated_U}
# time = {'time': time_step.reshape(-1,1)}
# savemat("interpolated_time_pseudo.mat", time)
# savemat("interpolated_u_pseudo.mat", u_dict)


### To play with the interpolated actions
env = Environment()
done = False
i = 0
x = []
y = []
z = []
act_state = env.state_space_to_state()
x1, y1, z1, v1, chi1, gamma1, uw1 = act_state
states_matrix = np.array([v1, gamma1, chi1, x1, y1, z1])
delta_change = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
minz = z1
while not done:
    actions = interpolated_U[i%len(time_step)] # mu, cl, T
    act_space = np.zeros(3)
    act_space[0] = env.cl_to_act_space(actions[1])
    act_space[1] = actions[0]/env.mu_scale
    act_space[2] = actions[2]/env.thrust_scale
    # print(actions)
    next_state, reward, done, _ = env.step(act_space)
    act_state = env.state_space_to_state()
    # x, y, z, V, chi, gamma, uw
    x1, y1, z1, v1, chi1, gamma1, uw1 = act_state
    states_matrix = np.vstack((states_matrix, np.array([v1, gamma1, chi1, x1, y1, z1])))

    dx, dy, dz, dv, dchi, dgamma = env.delta_change
    delta_change = np.vstack((delta_change, 100*np.array([dv, dgamma, dchi, dx, dy, dz])))
    x.append(act_state[0])
    y.append(act_state[1])
    z.append(act_state[2])
    print([x1, y1, z1])
    minz = min(minz, z1)
    i += 1
print(i)
print(z1)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


mat = loadmat('/home/suneet/Desktop/RL/project/Environment-Dynamic-Soaring/DynamicSoaring/envs/time_march.mat')
x1 = mat['x_tmarch']
y1 = mat['y_tmarch']
z1 = mat['z_tmarch']
# for i in range(len(x)):
#     print(x[i], y[i], z[i])
# ax.plot3D(x1, y1, z1, label="matlab")
ax.plot3D(x, y, z, label="python")
plt.legend()
plt.show()

# state_dict = {'states': states_matrix}
# savemat("states.mat", state_dict)

# delta_dict = {'delta_change': delta_change}
# savemat("delta_change.mat", delta_dict)