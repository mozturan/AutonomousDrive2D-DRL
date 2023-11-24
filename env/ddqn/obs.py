#observations shape
import numpy as np
import math

class observation_shape:
        def __init__(self, obs, info, num_history):
            self.num_history = num_history
            self.observation_shape = obs.shape
            infos = np.array([info["speed"], info["action"][0]], dtype=np.float16)
            self.info_shape = infos.shape
            relative = np.array([0.0,0.0],dtype= np.float16)
            self.relative_shape = relative.shape
            
        def info_edit(self, info):
            return np.array([info["speed"], info["action"][0]])
                            
        def obs_edit(self, observation):

            coordinates = [observation[1,1],observation[1,2]]
            relative_distance = sum(coordinate**2 for coordinate in coordinates)
            relative_distance = math.sqrt(relative_distance)

            relative_angle = math.atan2(coordinates[1], coordinates[0])
            relative = np.array([relative_distance, relative_angle])
            
            return relative
        
        def reset(self):
            
            self.last_observations = [np.zeros(self.observation_shape)] * self.num_history
            self.last_info = [np.zeros(self.info_shape)] * self.num_history
            self.last_relative = [np.zeros(self.relative_shape)] * self.num_history

        def update_input(self, obs, info):

            info  = self.info_edit(info)
            relative = self.obs_edit(obs)


            self.last_observations.append(obs)
            self.last_observations.pop(0)

            self.last_info.append(info)
            self.last_info.pop(0)

            self.last_relative.append(relative) # type: ignore
            self.last_relative.pop(0)

        def get_input(self):

            input = np.concatenate([self.last_observations[0].flatten(),
                                    self.last_observations[1].flatten(),
                                    self.last_relative[0].flatten(),
                                    self.last_relative[1].flatten(),
                                    self.last_info[0].flatten(),
                                    self.last_info[1].flatten()])


            return input

