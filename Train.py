import rl_utils
from CDQ import A2C
from tqdm import *
import os
import torch
import time
import matplotlib.pyplot as plt
from env2048.env import envs
from Train_off_policy import train_off_policy
from Train_on_policy import train_on_policy

if __name__ == '__main__':
    import argparse

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    parser = argparse.ArgumentParser(
        description=__doc__)
    # 训练设备:R7-5800H  RTX-3070-8G  16G内存 windows11
    # 训练设备类型
    # 保存的权重参数命名规则 main_net+轮次
    parser.add_argument('--resume', default=0, type=int,
                        help='resume from checkpoint')
    parser.add_argument('--entropy_discount_factor', default=1, type=int,
                        help='resume from checkpoint')
    #一开始可以将折算因子降低到0.95 后续再慢慢升高到0.995
    parser.add_argument('--gama', default=0.99, type=float, help='折算因子 discount_factor')
    # 训练的总轮次
    parser.add_argument('--epochs', default=650000, type=int, metavar='N',
                        help='number of total epochs to run')
    # 学习率
    #actor学习率设为1e-5 可以尝试更低的
    parser.add_argument('--p_lr', default=0.00001, type=float,
                         help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--v_lr', default=0.002, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')


    # 目标网络更新频率
    parser.add_argument('--target_update', default=5, type=int, help='目标网络的频率更新')
    args = parser.parse_args()
    train_on_policy(args)
