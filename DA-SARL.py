import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DAEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    # metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(4, ), dtype=np.float32)
        self.FiT = 0.04
        self.ToU = np.zeros((24)) + 0.08; self.ToU[9:17] = 0.13; self.ToU[17:21] = 0.18
    

    def step(self, action):
        #Start from 0h
        time = 0

        buyer_set = []
        seller_set = []

        #Sort order book
        buyer_set_sorted = sorted(buyer_set, key=lambda a: a[0], reverse=True)
        seller_set_sorted = sorted(seller_set, key=lambda a: a[0], reverse=False)

        #Double Auction
        i = 1
        j = 1
        while buyer_set_sorted[i][0] > seller_set_sorted[j][0]:
            q_trade = min(buyer_set_sorted[i][1], seller_set_sorted[j][1])
            p_trade = (buyer_set_sorted[i][0] + seller_set_sorted[j][0])/2

            if buyer_set_sorted[i][1] < q_trade:
                buyer_set_sorted[i][1] = 0
                i += 1
                seller_set_sorted[j][1] -= q_trade

            if seller_set_sorted[j][1] < q_trade:
                seller_set_sorted[j][1] = 0
                j += 1
                buyer_set_sorted[i][1] -= q_trade

            if i > len(buyer_set_sorted) or j > len(seller_set_sorted):
                break
        
            
        return observation, reward, done, info

    def reset(self, seed=None, options=None):
        ...
        return observation, info

    def render(self):
        ...

    def close(self):
        ...