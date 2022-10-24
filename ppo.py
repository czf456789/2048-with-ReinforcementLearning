import random

import torch
from Main_Net import Actor_Net as PolicyNet, Critic_Net as ValueNet
import rl_utils
import numpy as np
import torch.nn.functional as F


class PPO:
    ''' PPO算法,采用截断方式 '''

    def __init__(self, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma):
        device = torch.device("cuda:0")
        self.actor = PolicyNet().to(device)
        self.critic = ValueNet().to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.actor_optimizer1 = torch.optim.SGD(self.actor.parameters(),
                                                 lr=0.001)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.exp=0.1
    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def step(self, state, valid_action):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        state = torch.unsqueeze(state, dim=1)
        probs = self.actor(state)
        m = False
        #print(probs)
        #valid_action记录这本状态下的合法操作
        for i in range(4):
            if valid_action[i] != 0:
                probs[0][i] = 0
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    #predict时调用这个方法
    def step_predict(self, state, valid_action):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        state = torch.unsqueeze(state, dim=1)
        probs = self.actor(state)
        print(state)
        for i in range(4):
            if valid_action[i] != 0:
                probs[0][i] = 0
        print(probs)
        action=probs.argmax()
        # action_dist = torch.distributions.Categorical(probs)
        # action = action_dist.sample()
        return action.item()

    def save_weight(self, epoch):
        torch.save(self.actor.state_dict(), './save_weight/actor_net{}.pth'.format(epoch))
        torch.save(self.critic.state_dict(), './save_weight/critic_net{}.pth'.format(epoch))

    def load_weight(self, epoch):
        print('开始训练{}'.format(epoch))
        self.actor.load_state_dict(torch.load('./save_weight/actor_net{}.pth'.format(epoch)))
        self.critic.load_state_dict(torch.load('./save_weight/critic_net{}.pth'.format(epoch)))


    def test2(self):
        return self.actor.state_dict(), self.critic.state_dict()


    def update(self, transition):
        # self.update_once(punish)
        states = torch.tensor(np.array(transition['state']), dtype=torch.float).to(self.device)
        states = torch.unsqueeze(states, dim=1)
        next_states = torch.tensor(np.array(transition['next_state']), dtype=torch.float).to(self.device)
        next_states = torch.unsqueeze(next_states, dim=1)
        rewards = torch.tensor(transition['reward'], dtype=torch.float).view(-1, 1).to(self.device)
        actions = torch.tensor(transition['action']).view(-1, 1).to(
            self.device)
        dones = torch.tensor(transition['done']).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * dones
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)

        # t1=self.actor(states)
        # # 熵正则
        # entropy_loss = torch.mean(torch.sum(self.actor(states) * torch.log(self.actor(states))))
        # self.actor_optimizer1.zero_grad()
        # entropy_loss.backward()
        # self.actor_optimizer1.step()
        # m=torch.mean(torch.sum(self.actor(states) * torch.log(self.actor(states))))
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        a_loss = 0
        c_loss = 0
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            # e的x次方
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            a_loss += actor_loss
            c_loss += critic_loss
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


        return c_loss / self.epochs, a_loss / self.epochs


if __name__ == '__main__':
    probs = torch.tensor(np.array([[0, 0, 0, 0.001]]))
    action_dist = torch.distributions.Categorical(probs)
    for i in range(1000):
        action = action_dist.sample()
        print(action.item())
