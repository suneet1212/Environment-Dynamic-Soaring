from DynamicSoaring.envs.env import Environment
from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
# from matplotlib import pyplot as plt
import numpy as np
import sys
from typing import TypeVar, Optional
import time
from stable_baselines3.common.utils import safe_mean
AgentSelf = TypeVar("AgentSelf", bound="Agent")
from datetime import datetime

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

# Parallel environments
# env = make_vec_env("CartPole-v1", n_envs=4)
env = Environment()
# env = DummyVecEnv([lambda: env])

# print("#########################")
# print("tensorboard is off")
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./runs", seed=2)
# model.learn(total_timesteps=25000000)
# model.save("./models/ppo_rewardIsDistance_fixedStart")

# del model # remove to demonstrate saving and loading
class Agent(PPO):
    def __init__(self):
        super().__init__("MlpPolicy", env, verbose=1, tensorboard_log="./runs")
        # self.model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./runs")
        self.steps = 0
    
    # def step_learn(self, total_timesteps, save_intervals = 200000):
    #     i = 0
    #     while i < total_timesteps/save_intervals:
    #         self.learn(total_timesteps=save_intervals)
    #         savedir = "./models/ppo_env_rectified/"
    #         filename = str(i+125)
    #         path = savedir+filename
    #         self.save(path)
    #         i += 1
    
    def learn_model(
        self: AgentSelf,
        total_timesteps: int,
	dirname: str,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "./runs/ppo_debug",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        save_intervals: int = 200000,
    ) -> AgentSelf:
        iteration = 0
        i = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()
            #### Number of timesteps at the end of train may not be an exact multiple of save_intervals
            if int(self.num_timesteps/save_intervals) > i:
                i = self.num_timesteps//save_intervals
                savedir = "./model/"+dirname
                filename = str(i)
                path = savedir+filename
                self.save(path)

        callback.on_training_end()

        savedir = "./models/"+dirname
        filename = "final"
        path = savedir+filename
        self.save(path)
        return self

now = datetime.now()
dt = now.strftime("%d%m%Y%H%M%S")
agent = Agent()
agent.learn_model(150000000,"ppo_shortened_40_"+dt ,save_intervals=200000)
# agent.learn_model(10000, save_intervals=3000)


#x = []
#y = []
##z = []
#plt.ion()

#fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=False ,sharey=False)

#obs = env.reset()
#x.append(obs[0])
#y.append(obs[1])
#z.append(obs[2])

#actions = []
#count = 1

#plt.show()
#while count <= 5:
#    action, _states = model.predict(obs)
#    obs, rewards, dones, info = env.step(action)
#    actions.append(action)
#    x.append(obs[0])
#    y.append(obs[1])
#    z.append(obs[2])

#    fig.canvas.draw()
#    ax1.plot(x, y)
#    ax1.set_title('XY')
#    ax2.plot(y, z)
#    ax2.set_title("YZ")
#    ax3.plot(z, x)
#    ax3.set_title("ZX")
#    fig.canvas.flush_events()
#    if done:
#        print(count)
#        print(actions)
#        count += 1
#        # ax.plot3D(x, y, z, 'gray')
#        plt.pause(10)
#        x = []
#        y = []
#        z = []
#        actions = []
#    # env.render()
 
# # looping
# for _ in range(50):
   
#     # updating the value of x and y
#     line1.set_xdata(x*_)
#     line1.set_ydata(y)
#     line1.set_zda
 
#     # re-drawing the figure
     
#     # to flush the GUI events
#     fig.canvas.flush_events()
#     time.sleep(0.1)
# plt.pause()
