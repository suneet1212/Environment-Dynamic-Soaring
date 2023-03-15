import gym
from gym import spaces
import numpy as np
from scipy.io import loadmat
import os

class WindProfile():
    '''
    p = exp power = 1/7
    href = 20
    vr = 4m/s to 10m/s 
    '''
    def __init__(self, vr) -> None:
        self.p = 1/7.0
        self.href = 20
        self.vr = vr

    def uw(self, h):
        if h <= 0:
            return -1
        p = pow(h/self.href,self.p)
        return self.vr*p

class Aircraft():
    def __init__(self) -> None:
        ## any aircraft params we need to take as input
        # initialize params
        '''
        CL = lift coeff -> lift increases with this
        CD = coeff of drag, increases with CL, parabolic wrt CL
        '''
        self.CD0 = 0.0173 ## 
        self.CD1 = -0.0337 ##
        self.K = 0.0531 ##
        self.S = 0.3 ## Area of the wing
        self.m = 3.5 ## Mass in kg
        self.b = 1.5 # Wing span
        self.cl_max = 1.2 # 

        # self.x = pos[0]
        # self.y = pos[1]
        # self.z = pos[2]

        # self.yaw = ang[0]
        # self.pitch = ang[1]
        # self.roll = ang[2]
    
    def calcCD(self, cl):
        '''
        Coeff of drag
        '''
        return self.CD0 + self.CD1*cl + self.K*cl*cl

    def calcL(self, rho, cl, v):
        # should V be in aircraft class or in Env class
        # maybe can shift all the aircraft values like position, vel etc to this class
        return 0.5*rho*self.S*cl*v*v

    def calcD(self, rho, cd, v):
        return 0.5*rho*self.S*cd*v*v

    def vstall(self, rho, g):
        return np.sqrt(2*self.m*g/(rho*self.S*self.cl_max))

    ## Any other functions reqd?

class Environment(gym.Env):
    '''
    Constraints -> 
    all continuous values
    CL = -0.2 to cl_max
    T = thrust = [-0.001 - 0.001]
    mu = -75 to +75 degrees = roll
    70 >= h > 0, 
    x,y centred around 0 assumption
    max velocity = 12.4755 check vstall in main.m
    gamma = -60 to +60
    chi = 0 to 90
    '''
    def __init__(self) -> None:
        super().__init__()
        
        ## other env params like gravity
        self.g = 9.81
        self.rho = 1.2256 # density of air

        mat = None
        curr_dir = os.getcwd()
        mat = loadmat(os.path.join(curr_dir, 'DynamicSoaring/envs/Pseudospectral_states_2.mat'))
        self.target_U = mat['U']
        self.target_VR = mat['VR']
        self.target_X = mat['X']
        permutation = [3,5,4,0,1,2]
        idx = np.empty_like(permutation)
        idx[permutation] = np.arange(len(permutation))
        self.target_X[:] = self.target_X[:, idx]
        self.target_tvec = mat['tvec']

        self.traj_size = self.target_X.shape[0]
        self.timeDiff()

        ## Wind profile
        self.wind = WindProfile(self.target_VR[0][0])

        # need to initialise aircraft
        self.aircraft = Aircraft()
        self.vstall = self.aircraft.vstall(self.rho, self.g)

        # self.xlow = -400
        # self.ylow = 50
        # self.zlow = 0
        # self.vlow = self.vstall
        # self.chilow = -180
        # self.gammalow = -60
        # self.uwlow = 0
        self.actual_low = np.min(self.target_X, axis=0)
        # print("ERROR, need to fix this, order wrong")
        self.actual_high = np.max(self.target_X, axis=0)
        self.actual_low = np.append(self.actual_low, self.wind.uw(self.actual_low[2]))
        self.actual_high = np.append(self.actual_high, self.wind.uw(self.actual_high[2]))

        # self.xhigh = -200
        # self.yhigh = 250
        # self.zhigh = 100
        # self.vhigh = 70
        # self.chihigh = 180
        # self.gammahigh = 60
        # self.uwhigh = 15

        # self.xlow = min(self.target_X[])
        
        self.state = self.reset()
        self.actual_state = self.state_space_to_state()
        # state = (x,y,z, v, chi, gamma, ux(z))

        # self.x_max = 50
        # self.y_max = 50
        # self.z_max = 100
        # self.v_max = 70
        # self.chi_max = 90
        # self.gamma_max = 60
        # self.uw_max = 15


        # self.state_low = np.array([-400, 50, 0, self.vstall, 0, -self.gamma_max, 0])
        # self.state_high = np.array([-200, 250, self.z_max, self.v_max, self.chi_max, self.gamma_max, self.uw_max])
        
        self.state_low = np.array([-1, -1, -1, -1, -1, -1, -1])
        self.state_high = np.array([1, 1, 1, 1, 1, 1, 1])

        self.observation_space = spaces.Box(self.state_low, self.state_high)

        self.cl_scale = 1
        self.mu_scale = 75
        self.thrust_scale = 0.001
        # keep it from -1 to 1 wherever possible, use scaling factor to fill in rest.
        self.action_space = spaces.Box(low=np.array([-1, -1, -1], dtype=np.float), high=np.array([1, 1, 1], dtype=np.float))


        # self.max_ep_len = self.target_X.shape[0] #####
        # self.stepCount = 1

        # some timestep is reqd, need to know the estimate
        self.timestep = 0.01 ##############################
        self.time = 0

        # print(mat.keys())
        # print(mat['U'].shape, mat['VR'].shape, mat['X'].shape, mat['tvec'].shape)

        self.overlaps = 0 # number of times curr_ind has crossed start_ind

    def state_space_to_state(self):
        '''
            varlow = actual low values of the variable\n
            function takes state value between -1,1 range to :\n
            variable goes from (-1,1) => (varlow, varhigh)
        '''
        # x = self.state[0]
        # x = (self.state[0]+1)*0.5*(self.xhigh - self.xlow) + self.xlow
        # # y = self.state[1]
        # y = (self.state[1]+1)*0.5*(self.yhigh - self.ylow) + self.ylow
        # # z = self.state[2]
        # z = (self.state[2]+1)*0.5*(self.zhigh - self.zlow) + self.zlow
        # # v = self.state[3]
        # v = (self.state[3]+1)*0.5*(self.vhigh - self.vlow) + self.vlow
        # # chi = self.state[4]
        # chi = (self.state[4]+1)*0.5*(self.chihigh - self.chilow) + self.chilow
        # # gamma = self.state[5]
        # gamma = (self.state[5]+1)*0.5*(self.gammahigh - self.gammalow) + self.gammalow
        # # uw = self.state[6]
        # uw = (self.state[6]+1)*0.5*(self.uwhigh - self.uwlow) + self.uwlow
        actual_state = (self.state+1)*0.5*(self.actual_high-self.actual_low) + self.actual_low
        return actual_state

    def state_to_state_space(self, state):
        '''
            state is actual values of state variables\n
            varlow = actual low values of the variable\n
            variable goes from (varlow, varhigh) -> (-1, 1)\n
        '''
        # newState = np.zeros((7), dtype=np.float32)
        # x = state[0]
        # newState[0] = ((x-self.xlow)/(self.xhigh - self.xlow))*2 - 1
        # y = state[1]
        # newState[1] = ((y-self.ylow)/(self.yhigh - self.ylow))*2 - 1
        # z = state[2]
        # newState[2] = ((z-self.zlow)/(self.zhigh - self.zlow))*2 - 1
        # v = state[3]
        # newState[3] = ((v-self.vlow)/(self.vhigh - self.vlow))*2 - 1
        # chi = state[4]
        # newState[4] = ((chi-self.chilow)/(self.chihigh - self.chilow))*2 - 1
        # gamma = state[5]
        # newState[5] = ((gamma-self.gammalow)/(self.gammahigh - self.gammalow))*2 - 1
        # uw = state[6]
        # newState[6] = ((uw-self.uwlow)/(self.uwhigh - self.uwlow))*2 - 1
        # newState = ((state-self.actual_low)/(self.actual_high-self.actual_low))*2 -1
        newState = (state-self.actual_low)
        newState /= (self.actual_high-self.actual_low)
        newState *= 2
        newState -= 1
        # return np.array([x,y,z,v,chi,gamma,uw])
        return newState

    def cl_to_act_space(self, cl):
        '''
            cl is actual cl value\n
            cl_max is the actual max value of cl\n
            (-0.2, clmax) -> (-1, 1)\n
        '''
        return ((cl+0.2)/(self.aircraft.cl_max+0.2))*2 - 1

    def cl_to_act_value(self, act):
        '''
            act is action value\n
            (-1, 1) -> (-0.2, self.aircraft.cl_max)\n
        '''
        return (act+1)*0.5*(self.aircraft.cl_max+0.2) - 0.2

    def timeDiff(self):
        '''
            take difference between the consecutive elements of target_tvec\n
            save in dtvec\n
            has one element less than target_tvec\n
        '''
        self.dtvec = []
        for i in range(self.traj_size-1):
            self.dtvec.append(self.target_tvec[i+1] - self.target_tvec[i])

    def reset(self):
        '''
            to reset the state and time\n
            also reset episode length\n

            sample random start index -> forced it to start at index 0\n
            
            initialize all the state variables to the same point\n
            since this is in the actual state values, change it to -1,1 space\n

        '''
        # find newState
        # how to initialise the states, is there any constraint for the reset state
        # x = (np.random.rand()-0.5)*2*self.x_max
        # y = (np.random.rand()-0.5)*2*self.y_max
        # z = np.random.rand()*self.z_max
        # v = self.vstall + np.random.rand()*(self.v_max-self.vstall)
        # chi_d = np.random.rand()*self.chi_max
        # gamma_d = (np.random.rand()-0.5)*2*self.gamma_max
        self.time = 0

        # self.start_ind = np.random.randint(0,(self.traj_size)-1)
        self.start_ind = 0
        self.currInd = self.start_ind
        self.end_ind = (self.start_ind - 1)%(self.traj_size-1) ## since last and first points are the same

        self.episodeLength = 0
        
        x = self.target_X[self.start_ind][0]
        y = self.target_X[self.start_ind][1]
        z = self.target_X[self.start_ind][2]
        v = self.target_X[self.start_ind][3]
        chi = self.target_X[self.start_ind][4]
        gamma = self.target_X[self.start_ind][5]
        uw = self.wind.uw(z)

        newState = np.array([x, y, z, v, chi, gamma, uw])
        self.state = self.state_to_state_space(newState)
        self.actual_state = self.state_space_to_state()
        
        return self.state
    
    def isDone(self):
        '''
            check if the state values is between -1, 1\n
            if looping of index completed then return true\n
        '''
        # stop  it after n steps
        x = self.state_space_to_state()
        # if(x[0] >= self.state_high[0] or x[0] <= self.state_low[0]):
        #     return True
        # if(x[1] >= self.state_high[1] or x[1] <= self.state_low[1]):
        #     return True
        if(x[2] < 0.5):
        # if(self.state[2] < self.state_low[2]):
            # print("crash after ", self.episodeLength, " steps, with height = ", self.state[2])
            print("Crashing")
            return True
        # if(x[3] >= self.state_high[3] or x[3] <= self.state_low[3]):
        #     return True
        # if(x[4] >= self.state_high[4] or x[4] <= self.state_low[4]):
        #     return True
        # if(x[5] >= self.state_high[5] or x[5] <= self.state_low[5]):
        #     return True
        
        # # to check for 2 whole loops
        # if(self.currInd == self.end_ind):
        #     # print(self.episodeLength)
        #     self.overlaps += 1
        #     if(self.overlaps == 2):
        #         print("Length of trajectory is max")
        #         return True

        # Instead check for fraction of loops
        if float(self.currInd)/self.end_ind >= 0.3:
            return True
        return False

    def getReward(self):
        '''
            find the distance between the actual point and where it is supposed to be\n
            add extra negative reward if the actual height is lower than the minimum ht
        '''
        reward = 0

        # get x,y,z,v,chi,gamma of pt on required trajectory at time t
        x = self.target_X[self.currInd][0]
        y = self.target_X[self.currInd][1]
        z = self.target_X[self.currInd][2]
        v = self.target_X[self.currInd][3]
        chi = self.target_X[self.currInd][4]
        gamma = self.target_X[self.currInd][5]

        # # The following code will convert the state to x,y,z and then find reward.

        # state = self.state_space_to_state()
        # reward += (state[0] - x)**2
        # reward += (state[1] - y)**2
        # reward += (state[2] - z)**2
        # # reward -= (state[3] - v)**2
        # # reward -= (state[4] - chi)**2
        # # reward -= (state[5] - gamma)**2

        # # convert the required state into the state space and find distance (normalised reward)
        reqd_state = self.state_to_state_space(np.array([x,y,z,v,chi,gamma, self.wind.uw(z)]))
        for i in range(6):
            # add the squared distance between the state values except for wind
            reward += (reqd_state[i]-self.state[i])**2

        reward = -np.sqrt(reward)
        if(self.state[2] <= self.state_low[2]):
            reward -= 1000
        # return -np.sqrt(reward)
        print(reward)
        return reward


    
    def step(self, action):
        '''
            increment episode length by 1
            action in action space values, 
            convert to actual values

            get the actual state from the current state values
            
            calculate CD, L and D from the actions and current velocity
            find the change in each variable and multiply with the timestep to get new state

            increment time, check if done, calculate reward
        '''
        # get CL, mu and T from actions
        # state = (x,y,z, v, chi, gamma, uw(z))
        # actions = [cl, mu, T]
        # self.currInd += 1
        # self.currInd %= (self.traj_size-1)
        self.episodeLength += 1

        ## from action space to actual values
        cl = self.cl_to_act_value(action[0])
        mu_r = action[1]*self.mu_scale
        T = action[2]*self.thrust_scale
        
        ## from state space to actual values
        x,y,z,v,chi_r,gamma_r,uw = self.state_space_to_state()

        ## find aircraft params for environment dynamics
        cd = self.aircraft.calcCD(cl)
        l = self.aircraft.calcL(self.rho, cl, v)
        d = self.aircraft.calcD(self.rho, cd, v)

        ## Finding the angles in radians
        # chi_r = chi*np.pi/180
        # gamma_r = gamma*np.pi/180
        # mu_r = mu_d*np.pi/180

        # calculate duw
        duw = (uw*self.wind.p/z)
        duw = self.wind.vr*self.wind.p*np.power((abs(z)/self.wind.href), self.wind.p)/z

        # find dstate and change state accordingly
        dv = T/self.aircraft.m
        dv -= self.g*np.sin(gamma_r)
        dv -= d/self.aircraft.m
        dv -= duw*np.sin(chi_r)*np.cos(gamma_r)*v*np.sin(gamma_r)
        dv = self.timestep*dv
        
        dgamma = (-self.g*np.cos(gamma_r) + duw*np.sin(chi_r)*np.sin(gamma_r)*v*np.sin(gamma_r))
        dgamma += (l/self.aircraft.m)*np.cos(mu_r)
        dgamma *= 1/v
        # dgamma *= 180/np.pi
        dgamma = self.timestep*dgamma

        dchi = (1/(v*np.cos(gamma_r))) 
        dchi *= (-duw*np.cos(chi_r)*v*np.sin(gamma_r) + l/self.aircraft.m * np.sin(mu_r))
        # dchi *= 180/np.pi
        dchi = self.timestep*dchi

        dx = v*np.cos(gamma_r)*np.sin(chi_r) + uw
        dx = self.timestep*dx
        dy = v*np.cos(gamma_r)*np.cos(chi_r)
        dy = self.timestep*dy
        dz = v*np.sin(gamma_r)
        dz = self.timestep*dz
        # print(self.vstall)
        # print(dx)
        # self.state = self.state+dstate*self.dtvec[self.currInd]
        # newUw = self.wind.uw(z+dz)
        newUw = uw+duw*dz
        # newUw = self.wind.uw(z+dz)
        self.delta_change = np.array([dx, dy, dz, dv, dchi, dgamma])

        self.actual_state = np.array([x+dx, y+dy, z+dz, v+dv, chi_r+dchi, gamma_r+dgamma, newUw])
        newState = self.state_to_state_space(self.actual_state)
        # self.state = self.state+dstate
        self.state = newState
        done = self.isDone()
        self.time += self.timestep
        reward = self.getReward()
        # print(reward)
        # newState = self.state.copy()
        if(self.time >= self.dtvec[self.currInd]):
            self.time -= self.dtvec[self.currInd]
            self.currInd += 1
            self.currInd %= (self.traj_size-1)
        # print(self.state.shape)
        return newState, reward, done, {}

        ## RK 4 for timesteps

    # def render(self):
        
