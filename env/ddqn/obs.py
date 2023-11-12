#observations shape
import numpy as np

class observation_shape:
        def __init__(self, obs, info, num_history):
            self.num_history = num_history
            self.observation_shape = obs.shape
            infos = np.array([info["speed"], info["action"][0]], dtype=np.float16)
            self.info_shape = infos.shape
            
        def info_edit(self, info):
            return np.array([info["speed"], info["action"][0]])
                            
        def reset(self):
            
            self.last_observations = [np.zeros(self.observation_shape)] * self.num_history
            self.last_info = [np.zeros(self.info_shape)] * self.num_history

        def update_input(self, obs, info):

            info  = self.info_edit(info)

            self.last_observations.append(obs)
            self.last_observations.pop(0)

            self.last_info.append(info)
            self.last_info.pop(0)

        def get_input(self):
            # obs_stack = np.stack(self.last_observations)
            # info_stack = np.stack(self.last_info)

            input = np.concatenate([self.last_observations[0].flatten(),
                                    self.last_observations[1].flatten(),
                                    self.last_info[0].flatten(),
                                    self.last_info[1].flatten()])


            return input

