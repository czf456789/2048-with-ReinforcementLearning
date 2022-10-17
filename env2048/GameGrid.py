import torch

from env2048 import constants as c
import random
from env2048 import logic

import numpy as np


def gen():
    return random.randint(0, c.GRID_LEN - 1)


def create_random_matrix():
    p = np.array([0, 0, 0.15, 0.15, 0.15, 0.2, 0.15, 0.1, 0.1, 0, 0])
    k = np.random.choice([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], p=p.ravel())
    mat = np.array(0)
    mat = np.resize(mat, (4, 4))
    for i in range(4):
        for j in range(4):
            if random.random() < 0.7:
                mat[i][j] = np.random.choice([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], p=p.ravel())
    return mat


def log2(matrix):
    m = np.array(0)
    m = np.resize(m, (len(matrix[0]), len(matrix[0])))
    for i in range(len(matrix[0])):
        for j in range(len(matrix[0])):
            if matrix[i][j] != 0:
                m[i][j] = np.log2(matrix[i][j])
    return m


class GameGrid():
    def __init__(self):
        self.commands = {
            c.KEY_UP: logic.up,
            c.KEY_DOWN: logic.down,
            c.KEY_LEFT: logic.left,
            c.KEY_RIGHT: logic.right,
            c.KEY_UP_ALT1: logic.up,
            c.KEY_DOWN_ALT1: logic.down,
            c.KEY_LEFT_ALT1: logic.left,
            c.KEY_RIGHT_ALT1: logic.right,
            c.KEY_UP_ALT2: logic.up,
            c.KEY_DOWN_ALT2: logic.down,
            c.KEY_LEFT_ALT2: logic.left,
            c.KEY_RIGHT_ALT2: logic.right,
        }
        self.matrix = logic.new_game(c.GRID_LEN)

    def reset(self):
        self.__init__()

    def get_max_point(self):
        return np.array(self.matrix).max()

    def get_total_point(self):
        return np.array(self.matrix).sum()

    def getstate(self):
        return log2(self.matrix)

    def check_state(self):
        if logic.game_state(self.matrix) == 'lose':
            return False
        return True

    # 测试这个动作是否合法
    def step_test(self, action):
        if action in self.commands:
            _, done, _ = self.commands[action](self.matrix)
        return done

    def step_action(self, action):
        InstantReward = 0
        key = action
        if key in self.commands:
            # done为False意思为做出的动作是无效动作
            self.matrix, done, InstantReward = self.commands[key](self.matrix)
            # 如果移动了最大数
            # print(1231)
            max_value = np.array(self.matrix).max()
            #这里尝试了惩罚移动最大数字加入到训练中 但效果不好
            # if (self.matrix[0][0] != max_value or self.matrix[0][3] != max_value or self.matrix[3][0] != max_value or
            #         self.matrix[3][3] != max_value):
            #     # print("移动了最大值")
            #     InstantReward -= (np.array(self.matrix).max()*4-self.matrix[0][0]-self.matrix[0][3]-self.matrix[3][0]-self.matrix[3][3])/ 100
            if logic.game_state(self.matrix) == 'lose':
                factor = 11 - np.log2(np.array(self.matrix).max())
                InstantReward = -50 * factor
                # print("游戏失败")
                return log2(self.matrix), InstantReward, False, done
            self.matrix = logic.add_two(self.matrix)
        return log2(self.matrix), float(InstantReward), True, done


if __name__ == '__main__':
    x = [[0, 3, 21, 34], [0, 3, 21, 38], [0, 3, 21, 38], [0, 3, 21, 38]]
    c = 34
    print(x[3][3])
    print(c in x[0])
    print(np.array(x).argmax())
