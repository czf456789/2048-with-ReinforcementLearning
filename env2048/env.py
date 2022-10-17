import random
from typing import Optional, Union, Tuple
from env2048 import constants as c

from env2048.GameGrid import GameGrid
import random

#封装的环境类
class envs():

    def __init__(self):
        self.game = GameGrid()
        self.state = self.game.getstate()

    def get_game_point(self):
        return self.game.get_max_point(), self.game.get_total_point()
    #试探四个动作的合法性
    def get_valid_action(self):
        actions = [c.KEY_UP, c.KEY_DOWN, c.KEY_LEFT, c.KEY_RIGHT]
        valid_action=[0,0,0,0]
        if self.game.check_state():
            for i in range(4):
                if not self.game.step_test(actions[i]):
                    valid_action[i]+=1
        return valid_action
    #用于训练
    def step(self, action):
        actions = [c.KEY_UP, c.KEY_DOWN, c.KEY_LEFT, c.KEY_RIGHT]
        state, reward, game_done ,is_legal= self.game.step_action(actions[action])
        if is_legal:
            #print("执行了非法操作")
            return state, reward, game_done
        self.state = state
        return state, reward, game_done


    def check_state(self):
        return self.game.check_state()
    # 重新生成棋盘
    def reset(self):
        self.game.reset()
        self.state = self.game.getstate()
        return self.state




if __name__ == '__main__':
    print(random.randint(0, 3))
