import torch

from env2048 import constants as c
import random
from env2048 import logic

import numpy as np


def log2(matrix):
    m = np.array(0)
    m = np.resize(m, (len(matrix[0]), len(matrix[0])))
    for i in range(len(matrix[0])):
        for j in range(len(matrix[0])):
            if matrix[i][j] != 0:
                m[i][j] = np.log2(matrix[i][j])
    return m


class GameGrid():
    def __init__(self,difficulty_factor):
        #四种动作
        self.commands = {
            logic.up,
            logic.down,
            logic.left,
            logic.right,
        }
        self.difficulty=difficulty_factor


    def reset(self):
        state = logic.new_game(c.GRID_LEN,self.difficulty)
        return state

    def check_state(self, state):
        if logic.game_state(state) == 'lose':
            return False
        return True

    # 测试这个动作是否合法
    def step_test(self, action, state):
        if action in self.commands:
            _, done, _ = self.commands[action](state)
        return done

    def get_valid_action(self, matrix):
        actions = [c.KEY_UP, c.KEY_DOWN, c.KEY_LEFT, c.KEY_RIGHT]
        valid_action = [0, 0, 0, 0]
        if self.check_state(matrix):
            for i in range(4):
                if not self.step_test(actions[i], matrix):
                    valid_action[i] += 1
        #返回值若 valid_action[i]=1 说明动作i是无效动作
        return valid_action

    def step_action(self, action, state):
        InstantReward = 0
            # done为False意思为做出的动作是无效动作
        next_state, done, InstantReward = self.commands[action](state)
        if logic.game_state(next_state) == 'lose':
                factor = 10 - np.log2(np.array(next_state).max())
                InstantReward = -20 * factor
                # print("游戏失败")
                return log2(next_state), InstantReward, False, done
        #在随机位置添加一个数字
        next_state = logic.add_two(next_state,self.difficulty)
        return log2(next_state), InstantReward, True, done
