import random
from typing import Optional, Union, Tuple
from env2048 import constants as c

from env2048.GameGrid import GameGrid
import random
import numpy as np


# 封装的环境类
class envs():

    def __init__(self):
        self.game = GameGrid()
        #一局游戏的矩阵
        self.matrix = self.game.reset()

    def get_game_point(self):
        return np.max(self.matrix), np.sum(self.matrix)

    # 试探四个动作的合法性
    def get_valid_action(self):
        return self.game.get_valid_action(self.matrix)

    def step(self, action):
        self.matrix, reward, game_done, is_legal = self.game.step_action(action, self.matrix)
        return self.matrix, reward, game_done

    def check_state(self):
        return self.game.check_state(self.matrix)

    # 重新生成棋盘
    def reset(self):
        self.matrix = self.game.reset()
        return self.matrix
