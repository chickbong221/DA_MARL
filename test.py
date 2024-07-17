import numpy as np

# x = [[3, 5], [1, 9], [4, 7], [2, 8]]
# print(np.zeros((24))+0.1)

# possible_agents = [f"consumer_{i+1}" for i in range(4)] + [f"prosumer_{i+1}" for i in range(4)]
# ES_begin = {
#     a: np.clip(np.random.normal(6,1), 2, 10) for a in possible_agents
# }
# print(ES_begin)

# def return_test():
#     return "ahihi"

# # for i in range(3):
# #     print(return_test())

# f = 1

# def big_func():
#     # f = 0
#     while f < 5:
#         f += 1
#         print(f"This is {f-1}")
#         print(return_test())

# big_func() 
# possible_agents = [f"consumer_{i+1}" for i in range(4)] + [f"prosumer_{i+1}" for i in range(4)]

# PV_generation = {
#     a: 0 if "consumer" in a else np.random.normal(15, 3) for a in possible_agents
# }
# for a in possible_agents:
#     print(PV_generation[a])

# possible_agents = [f"consumer_{i+1}" for i in range(4)] + [f"prosumer_{i+1}" for i in range(4)]

# for agent in possible_agents:
#     ahihi = {a: (agent, 0.1, 0.9) for a in range(3)}
# print(ahihi[1][0])

# a = { 
#     "cam": []
#     }
# for i in range(3):
#     a["cam"].append(i)
# print(a["cam"])

# a = [[1,2],[2,4],[6,3]]
# buyer_set_sorted = sorted(a, key=lambda a: a[1], reverse=True)
# print(buyer_set_sorted)

a=10
print(-a)