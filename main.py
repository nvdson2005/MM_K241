import gymnasium as gym
import gym_cutting_stock
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx

import numpy as np
import time
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
    observation, info = env.reset(seed=42)
    policy2210xxx = Policy2210xxx(policy_id=1)
    for _ in range(200):
        action = policy2210xxx.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        print()
        print(_, "st loop: ", info)
        #print(action)
        time.sleep(0.25)
        if terminated or truncated:
            observation, info = env.reset()

env.close()
