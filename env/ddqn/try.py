import gymnasium as gym
import numpy as np
import pprint
# from utils import record_videos, load_config
import highway_env

# data = load_config()
env = gym.make('racetrack-v0', render_mode = 'rgb_array')
# env.configure(data) # type: ignore
# pprint.pprint(env.config) # type: ignore

for i in range(10):
    (obs, info), done = env.reset(), False
    env.render()

    while not done:
            
            new_observation, reward, done, truncated, new_info = env.step(action=[-0.2])
