import gymnasium as gym
import gym_cutting_stock
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2312900_2310559_2420003_2312894_2312974
import numpy as np

import time
#from student_submissions.s2210xxx.SimulatedAnnealing import SimulatedAnnealingPolicy
# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 2

if __name__ == "__main__":
    # Reset the environment
    # observation, info = env.reset(seed=42)
    # #print(info)
    # print(observation["stocks"])
    # # Test GreedyPolicy
    # gd_policy = GreedyPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = gd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     print(ep, ": ", info)
    #     print()
    #     if terminated or truncated:
    #         print(info)
    #         observation, info = env.reset(seed=ep)
    #         ep += 1

    # # Reset the environment
    # observation, info = env.reset(seed=42)

    # # Test RandomPolicy
    # rd_policy = RandomPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = rd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     print(ep, ": ", info)
    #     print()
    #     if terminated or truncated:
    #         print(info)
    #         observation, info = env.reset(seed=ep)
    #         ep += 1

    #Uncomment the following code to test your policy
    # Reset the environment
    
    #simulated_annealing_policy = SimulatedAnnealingPolicy()
    # for _ in range(200):
    
    #This is used for testing one single list of orders
    # products = observation["products"]
    # _ = 0
    # while all([product["quantity"] == 0 for product in products]) == 0:
    #     action = policy2210xxx.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     print()
    #     print(_, "st loop: ", info)
    #     _ += 1
    #     #time.sleep(0.5)
    #     #print(action)
    #     if terminated or truncated:
    #         print("===========================")
    #         print("Finished final order. Final Result")
    #         print(info)
    #         time.sleep(10)
    
    observation, info = env.reset(seed=42)
    policy2210xxx = Policy2312900_2310559_2420003_2312894_2312974(policy_id=1)
    # This is used for testing multiple orders. When the loop time is less than 200, if the list of orders is finished,
    # It gets a new order using reset and continue to test the policy.
    print("First Fit Decreasing implementation:")
    print("=====================================")
    print()
    time.sleep(3)
    for _ in range(200):
        action = policy2210xxx.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        print()
        print(_, "st loop: ", info)
        #print(action)
        if terminated or truncated:
            print("===========================")
            if _ < 199:
                print("Finished order. Result: ")
                print(info)
                print("Getting new order in five seconds...")
                time.sleep(5)
                observation, info = env.reset()
            else:
                print("Finished final order. Final Result")
                print(info)
                time.sleep(5)
                observation, info = env.reset()
    
    observation, info = env.reset(seed=42)        
    policy2210xxx = Policy2312900_2310559_2420003_2312894_2312974(policy_id=2)
    print("Simulated Annealing implementation:")
    print("=====================================")
    print()
    time.sleep(3)
    for _ in range(200):
        action = policy2210xxx.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        print()
        print(_, "st loop: ", info)
        #print(action)
        if terminated or truncated:
            print("===========================")
            if _ < 199:
                print("Finished order. Result: ")
                print(info)
                print("Getting new order in five seconds...")
                time.sleep(5)
                observation, info = env.reset()
            else:
                print("Finished final order. Final Result")
                print(info)
                time.sleep(5)
                observation, info = env.reset()

env.close()
