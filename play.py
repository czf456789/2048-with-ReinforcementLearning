import threading
from tkinter import Frame, Label, CENTER, Button
import random
from env2048 import logic
from env2048 import constants as c
import sys
from ppo import PPO
import time
import torch
import numpy as np

import threading


def gen():
    return random.randint(0, c.GRID_LEN - 1)


def log2(matrix):
    m = np.array(0)
    m = np.resize(m, (len(matrix[0]), len(matrix[0])))
    for i in range(len(matrix[0])):
        for j in range(len(matrix[0])):
            if matrix[i][j] != 0:
                m[i][j] = np.log2(matrix[i][j])
    return m


class GameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)
        self.agent = PPO(0, 0, 0, 0, 0, 0)
        self.agent.load_weight(327600)
        self.agent.eval()
        self.grid()
        self.master.title('2048')
        # 若需使用搜索树 则将参数改为self.mcts_action
        self.master.bind("<Key>", self.mcts_action)
        self.TotalReward = 0
        self.InstantReward = 0
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

        self.grid_cells = []
        self.matrix = logic.new_game(c.GRID_LEN)
        self.init_grid()
        self.start = False
        self.history_matrixs = []
        self.update_grid_cells()

    def getstate(self):
        return self.matrix

    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME, width=c.SIZE, height=c.SIZE)
        # 展示视图
        background.grid()
        for i in range(c.GRID_LEN):
            # 一行一行渲染
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(
                    background,
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    width=c.SIZE / c.GRID_LEN,
                    height=c.SIZE / c.GRID_LEN
                )
                cell.grid(
                    row=i,
                    column=j,
                    padx=c.GRID_PADDING,
                    pady=c.GRID_PADDING
                )
                t = Label(
                    master=cell,
                    text="",
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    justify=CENTER,
                    font=c.FONT,
                    width=5,
                    height=2)
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)

    # 这里是渲染工作
    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(
                        text=str(new_number),
                        bg=c.BACKGROUND_COLOR_DICT[new_number],
                        fg=c.CELL_COLOR_DICT[new_number]
                    )
                    # 更新窗口
        self.update_idletasks()

    def get_valid_action(self):
        actions = [c.KEY_UP, c.KEY_DOWN, c.KEY_LEFT, c.KEY_RIGHT]
        valid_action = [0, 0, 0, 0]
        if logic.game_state(self.matrix):
            for i in range(4):
                _, done, _ = self.commands[actions[i]](self.matrix)
                if not done:
                    valid_action[i] += 1
        return valid_action

    def get_valid_actions(self, state):
        actions = [c.KEY_UP, c.KEY_DOWN, c.KEY_LEFT, c.KEY_RIGHT]
        valid_action = [0, 0, 0, 0]
        if logic.game_state(state):
            for i in range(4):
                _, done, _ = self.commands[actions[i]](state)
                if not done:
                    valid_action[i] += 1
        return valid_action

    def reinforce(self, event):
        self.InstantReward = 0
        key = event.keysym
        if self.start:
            return
        self.start = True
        actions = [c.KEY_UP, c.KEY_DOWN, c.KEY_LEFT, c.KEY_RIGHT]
        while not logic.game_state(self.matrix) == 'lose':
            valid_action = self.get_valid_action()
            action = self.agent.step_predict(log2(self.matrix), valid_action)

            self.matrix, done, self.InstantReward = self.commands[actions[action]](self.matrix)
            self.TotalReward += self.InstantReward
            if done:
                self.matrix = logic.add_two(self.matrix)
                self.update_grid_cells()
                # 开始检查状态
                if logic.game_state(self.matrix) == 'win':
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                if logic.game_state(self.matrix) == 'lose':
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.InstantReward = -sys.maxint - 1
                    self.TotalReward += self.InstantReward

    def mcts_action(self, event):
        key = event.keysym
        actions = [c.KEY_UP, c.KEY_DOWN, c.KEY_LEFT, c.KEY_RIGHT]
        if self.start:
            return
        self.start = True
        while not logic.game_state(self.matrix) == 'lose':
            # 以下是使用搜索树的方法
            value = [0, 0, 0, 0]
            self.mcts(self.matrix, value, 0, -1)
            # 如果搜索失败就采取随即动作
            if np.sum(value) == 0:
                valid_action = self.get_valid_action()
                action = random.randint(0, 3)
                while valid_action[action] != 0:
                    action = random.randint(0, 3)
            else:
                action = np.argmax(value)
            self.matrix, done, self.InstantReward = self.commands[actions[action]](self.matrix)
            if done:
                self.matrix = logic.add_two(self.matrix)
                self.update_grid_cells()
                # 开始检查状态
                if logic.game_state(self.matrix) == 'win':
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                if logic.game_state(self.matrix) == 'lose':
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.InstantReward = -sys.maxint - 1
                    self.TotalReward += self.InstantReward

    def mcts(self, state, value, deep, father):
        actions = [c.KEY_UP, c.KEY_DOWN, c.KEY_LEFT, c.KEY_RIGHT]
        valid_action = self.get_valid_actions(state)
        if deep == 0:
            for i in range(4):
                if valid_action[i] == 0:
                    next_state, done, reward = self.commands[actions[i]](state)
                    father = i
                    self.mcts(next_state, value, deep + 1, father)
        if deep < 6 and deep != 0:
            for i in range(4):
                if valid_action[i] == 0:
                    next_state, done, reward = self.commands[actions[i]](state)
                    _, empty_cells = logic.check_Is_full(next_state)
                    empty_cells = empty_cells / 16
                    if done:
                        next_state = logic.add_two(next_state)
                        # 开始检查状态
                        if logic.game_state(next_state) == 'lose':
                            return
                    value[father] += (empty_cells + reward) * pow(0.95, deep)
                    self.mcts(next_state, value, deep + 1, father)


def generate_next(self):
    index = (gen(), gen())
    while self.matrix[index[0]][index[1]] != 0:
        index = (gen(), gen())
    self.matrix[index[0]][index[1]] = 2


# 运行
if __name__ == '__main__':
    x = GameGrid()
    x.mainloop()
