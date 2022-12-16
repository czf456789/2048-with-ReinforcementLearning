import os
import torch
import time
import matplotlib.pyplot as plt
import rl_utils
from ppo import PPO
from tqdm import *
from env2048.env import envs
import numpy as np


def train_on_policy(args):
    cpu_device = torch.device('cpu')
    Train_state = "Train"
    print(args)
    actor_lr = args.p_lr
    critic_lr = args.v_lr
    num_episodes = args.epochs
    lmbda = 0.98
    epochs = 10
    gamma = args.gama
    eps = 0.2
    agent = PPO(actor_lr, critic_lr, lmbda, epochs, eps, gamma)
    env2048 = envs(args.difficulty_factor)
    return_list = []
    critic_loss_list = []
    agent.train()
    actor_loss_list = []
    if args.resume != 0:
        agent.load_weight(args.resume)
    start_epoch = args.resume
    # 500回合的平均回报
    mean_reward = 0
    max_100_point = 0
    transition = {'state': [], 'action': [], 'next_state': [], 'reward': [], 'done': []}
    for i in range(int(start_epoch / 300), int(num_episodes / 300)):
        max_100_point = 0
        max_total_point = 0
        with tqdm(total=300, desc='episode %d' % i) as pbar:
            for i_episode in range(300):
                state = env2048.reset()
                done = True
                while done:
                    # 无效动作在step之前以及提前过滤掉了

                    valid_action = env2048.get_valid_action()
                    # 如果是从过滤非法操作中采样出来的，给与更高的奖励
                    action= agent.step(state, valid_action)

                    next_state, reward, done = env2048.step(action)
                    transition['state'].append(state)
                    transition['action'].append(action)
                    transition['next_state'].append(next_state)
                    transition['reward'].append(reward)
                    transition['done'].append(int(done))
                    state = next_state
                max_point, total_point = env2048.get_game_point()
                if max_point > max_100_point:
                    max_100_point = max_point
                if total_point > max_total_point:
                    max_total_point = total_point
                if i_episode % 5 == 0 and i_episode != 0:
                    critic_loss, actor_loss = agent.update(transition)
                    critic_loss_list.append(critic_loss.to(cpu_device).detach())
                    if actor_loss != 0:
                        actor_loss_list.append(actor_loss.to(cpu_device).detach())
                    transition = {'state': [], 'action': [], 'next_state': [], 'reward': [], 'done': []}

                return_list.append(total_point)
                mean_reward +=total_point
                pbar.set_postfix({
                    'mean_return':
                        '%.5f' % ((mean_reward) / (i_episode + 1)),
                    'max_point':
                        '%d' % max_point,
                    'total_point':
                        '%d' % total_point,
                    'max_100_point': "{}".format(max_100_point),
                    'max_total_point': "{}".format(max_total_point),
                    'time':
                        '{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                })
                pbar.update(1)
            agent.save_weight((i + 1) * 300)
            plot('return', return_list, i * 300)
            plot('critic', critic_loss_list, i * 300)
            plot('actor', actor_loss_list, i * 300)
            mean_reward = 0


def plot(type, data_list, path_name):
    episodes_list = list(range(len(data_list)))
    mv_return = rl_utils.moving_average(data_list, 5)
    plt.cla()
    plt.plot(episodes_list, mv_return)
    plt.title('PPo on 2048')
    plt.xlabel('Episodes')
    if type == 'return':
        plt.ylabel('Returns')
    elif type == 'critic':
        plt.ylabel('critic_loss')
    else:
        plt.ylabel('actor_loss')
    plt.savefig('./Train_result/{}.jpg'.format(type))
