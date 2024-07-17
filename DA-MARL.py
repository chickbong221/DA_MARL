import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Box

from pettingzoo import ParallelEnv


class CustomEnvironment(ParallelEnv):

    metadata = {
        "name": "custom_environment",
    }

    def __init__(self):
       
        #agent_list
        self.possible_agents = [f"consumer_{i+1}" for i in range(4)] + [f"prosumer_{i+1}" for i in range(4)]

        #grid price
        self.FiT = 0.04
        self.ToU = np.zeros((24)) + 0.08; self.ToU[9:17] = 0.13; self.ToU[17:21] = 0.18

        #ES properties
        self.ES_maximum_energy_level = 10
        self.ES_minimum_energy_level = 2
        self.ES_power_capacity = 2
        self.ES_charging_efficiency = 0.95
        self.ES_discharging_efficiency = 0.95
        self.ES_energy_level = {a: np.clip(np.random.normal(6,1), 2, 10) for a in self.possible_agents}

    def reset(self, seed=None, options=None):

        if seed is not None:
            np.random.seed(seed)
        
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        # self.ES_energy_level = {a: np.clip(np.random.normal(6,1), 2, 10) for a in self.agents}
        self.inflexible_demand = {a: min(0, np.random.normal(7, 3)) for a in self.agents}
        self.PV_generation = {a: 0 if "consumer" in a else np.random.normal(12, 3) for a in self.agents}

        observations = {
            a: (
                self.inflexible_demand[a] - self.PV_generation[a],
                self.ES_energy_level[a],
                self.FiT,
                self.ToU[self.timestep],
            )
            for a in self.agents
        }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):

        self.shared_reward = 0
        self.timestep += 1
        self.inflexible_demand = {a: min(0, np.random.normal(6, 2)) for a in self.agents}
        self.PV_generation = {a: 0 if "consumer" in a else np.random.normal(11, 2) for a in self.agents}

        # Execute actions and get trading information
        buyer_set = []
        seller_set = []
        for agent in self.agents:
            
            # Calculate quantity for trading
            inflexible_load = self.inflexible_demand[agent] - self.PV_generation[agent]
            current_ES_energy_level = self.ES_energy_level[agent]
            quantity_for_ES = actions[agent][0]
            # print(quantity_for_ES) #debug

            if quantity_for_ES >= 0:
                C = min(quantity_for_ES * self.ES_power_capacity, (self.ES_maximum_energy_level - current_ES_energy_level) / self.ES_charging_efficiency)
                D = 0
            if quantity_for_ES < 0:
                C = 0
                D = min(quantity_for_ES*self.ES_power_capacity, (self.ES_minimum_energy_level - current_ES_energy_level) * self.ES_discharging_efficiency)
            quantity_final = inflexible_load + C + D
            self.ES_energy_level[agent] += (C*self.ES_charging_efficiency + D*self.ES_discharging_efficiency)


            # Calculate price for trading
            price_selection = actions[agent][1]
            # print(price_selection)

            price_final = self.FiT + price_selection * (self.ToU[self.timestep] - self.FiT) 
            # print(price_final)

            # Apply abs() before go in DA algorithm
            if quantity_final >= 0:
                buyer_set.append([agent, np.abs(quantity_final), price_final])
            else:
                seller_set.append([agent, np.abs(quantity_final), price_final])
            print(buyer_set)

        #Sort order book
        buyer_set_sorted = sorted(buyer_set, key=lambda a: a[2], reverse=True)
        seller_set_sorted = sorted(seller_set, key=lambda a: a[2], reverse=False)
        # a = [u.shape for u in seller_set_sorted]
        # print(a)

        #Double Auction
        trading_history = {a: [] for a in self.agents}
        i = 0
        j = 0
        # print(buyer_set_sorted[i][2], seller_set_sorted[j][2] )
        while buyer_set_sorted[i][2] > seller_set_sorted[j][2]:
            q_trade = min(buyer_set_sorted[i][1], seller_set_sorted[j][1])
            p_trade = (buyer_set_sorted[i][2] + seller_set_sorted[j][2])/2

            if buyer_set_sorted[i][1] <= q_trade:
                buyer_set_sorted[i][1] = 0
                seller_set_sorted[j][1] -= q_trade

                trading_history[buyer_set_sorted[i][0]].append([q_trade, p_trade])
                trading_history[seller_set_sorted[j][0]].append([-q_trade, p_trade])
                i += 1

            if seller_set_sorted[j][1] <= q_trade:
                seller_set_sorted[j][1] = 0
                buyer_set_sorted[i][1] -= q_trade

                trading_history[buyer_set_sorted[i][0]].append([q_trade, p_trade])
                trading_history[seller_set_sorted[j][0]].append([-q_trade, p_trade])
                j += 1

            if i > (len(buyer_set_sorted) - 1) or j > (len(seller_set_sorted) - 1):
                break

        for i in range(len(buyer_set_sorted)):
            if buyer_set_sorted[i][1] != 0:
                trading_history[buyer_set_sorted[i][0]].append([buyer_set_sorted[i][1], self.ToU[self.timestep]]) 
                buyer_set_sorted[i][1] = 0
        for j in range(len(seller_set_sorted)):
            if seller_set_sorted[j][1] != 0:
                trading_history[seller_set_sorted[j][0]].append([-seller_set_sorted[j][1], self.FiT])
                seller_set_sorted[j][1] = 0

        for agent in self.agents:
            for k in range(len(trading_history[agent])):
                self.shared_reward += trading_history[agent][k][0]*trading_history[agent][k][1]
        self.shared_reward /= len(self.agents)

        rewards = {a: self.shared_reward for a in self.agents}

        # Check termination conditions
        terminations = {a: False for a in self.agents}

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep == 23:
            truncations = {a: True for a in self.agents}
        
        # Get observations
        observations = {
            a: (
                self.inflexible_demand[a] - self.PV_generation[a],
                self.ES_energy_level[a],
                self.FiT,
                self.ToU[self.timestep],
            )
            for a in self.agents
        }

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        self.timestep += 1

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Renders the environment."""
        
    #observation space
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    #action space
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        low = np.array([-1, 0])
        high = np.array([1, 1])
        return Box(low=low, high=high, dtype=np.float32)
    

from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = CustomEnvironment()
    parallel_api_test(env, num_cycles=1_000_000)
