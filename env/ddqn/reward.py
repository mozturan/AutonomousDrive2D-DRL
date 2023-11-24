import numpy as np

def _reward(info, observation):

    """
    *Reward = Lane_centering + Collision + Alive + Distance (NORMALIZED)
    """
    rewards = info['rewards'] #: {'lane_centering_reward': 0.13575132365757345, 'action_reward': False, 'collision_reward': True, 'on_road_reward': True}
    alive = 1.0
    lidar_detect = observation[1,0]
    relative_distance = [observation[1,1],observation[1,2]]

    if lidar_detect != 0:
        distance_reward = sum(coordinate**2 for coordinate in relative_distance)
        # distance_reward = 1-np.sqrt(distance_reward)

        distance_reward = 1/np.sqrt(distance_reward)
    else:
        distance_reward = 0

    if not rewards["on_road_reward"]:
        reward = -1.
    elif rewards["collision_reward"]:
        reward = -1. 
    else:
        reward = rewards["lane_centering_reward"] + (0.3*alive) - (0.1*distance_reward)

    # config = {
    #     "collision_reward": -1.0,
    #     "lane_centering_reward" : 1.0,
    #     # "on_road_reward" : 0.1,
    #     "action_reward" : 0.1         
    #             }
    
    # reward_ = sum(rewards.get(name, 0) * reward for name, reward in config.items())


    # reward = reward_ - (0.3* distance_reward)
    # reward = (reward+1)/2

    return reward