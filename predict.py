import random

from ppo import PPO
from env2048.env import envs
import torch
import numpy as np

#测试一百轮的所有得分

def predict():
    agent = PPO(0, 0, 0, 0, 0, 0)
    env = envs()
    agent.load_weight(327600)
    agent.eval()
    state = env.reset()
    punish = [0, 0, 0, 0]
    done = True
    total_reward = []
    max_p = []
    max_t_p = []
    show = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    e_return = 0
    total_count=0
    punish_count=0
    counts=0
    for i in range(100):
        e_return = 0
        state = env.reset()
        done = True
        while done:
            valid_action = env.get_valid_action()
            total_count+=1
            action = agent.step_predict(state,valid_action)
            next_state, reward, done = env.step(action)
            if reward > 0:
                e_return += reward
            state=next_state

        total_reward.append(e_return)
        max_point, max_total_point = env.get_game_point()
        count[int(np.log2(max_point))] += 1
        max_p.append(max_point)
        max_t_p.append(max_total_point)
    print(show)
    print(count)
    print(punish_count/total_count)
    return total_reward, max_p, max_t_p





if __name__ == '__main__':
    total_reward, max_p, max_t_p = predict()
    print(total_reward)
    print(max_p)
    print(max_t_p)

    #test()
    # device=torch.device("cuda:0")
    # a = torch.ones((2, 2), requires_grad=True)
    # c = a.to(device)
    #
    # b = c.sum()
    # b.backward()
    #
    # print(a.grad)
