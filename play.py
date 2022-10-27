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
import matplotlib

matplotlib.use('Agg')
import threading


def log2(matrix):
    m = np.array(0)
    m = np.resize(m, (len(matrix[0]), len(matrix[0])))
    for i in range(len(matrix[0])):
        for j in range(len(matrix[0])):
            if matrix[i][j] != 0:
                m[i][j] = np.log2(matrix[i][j])
    return m


class GameGrid(Frame):
    def __init__(self, difficulty):
        self.f = Frame.__init__(self)
        self.show = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        self.count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.difficulty = difficulty
        self.commands = [logic.up,
                         logic.down,
                         logic.left,
                         logic.right]
        self.matrix = logic.new_game(c.GRID_LEN, self.difficulty)
        self.test_epoch = 0

    def enable_graph(self):
        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.mcts_action)
        self.grid_cells = []
        self.init_grid()
        self.update_grid_cells()
        self.mainloop()

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

    def get_valid_action(self, state):
        valid_action = [0, 0, 0, 0]
        if logic.game_state(state) != 'lose':
            for action in range(4):
                _, done, _ = self.commands[action](state)
                if not done:
                    valid_action[action] += 1
        return valid_action

    def destroy_window(self):
        Frame.destroy(self)
        Frame.quit(self)

    def reinforce(self, event):
        for i in range(self.test_epoch):
            self.matrix = logic.new_game(c.GRID_LEN, self.difficulty)
            while not logic.game_state(self.matrix) == 'lose':
                valid_action = self.get_valid_action()
                action = self.agent.step_predict(log2(self.matrix), valid_action)
                self.take_action(action)
            max_point = np.max(self.matrix)
            max_total_point = np.sum(self.matrix)
            self.count[int(np.log2(max_point))] += 1
            time.sleep(1)
        print(show)
        print(count)
        self.destroy_window()

    def mcts_action(self, event):
        for i in range(self.test_epoch):
            self.matrix = logic.new_game(c.GRID_LEN, self.difficulty)
            self.update_grid_cells()
            while not logic.game_state(self.matrix) == 'lose':
                # 以下是使用搜索树的方法
                value = [0, 0, 0, 0]
                self.mcts(self.matrix, value, 0, -1)
                action = np.argmax(value)
                self.take_action(action)
            max_point = np.max(self.matrix)
            max_total_point = np.sum(self.matrix)
            self.count[int(np.log2(max_point))] += 1
            time.sleep(2)
        print(self.show)
        print(self.count)
        self.destroy_window()

    def mcts(self, state, value, deep, father):
        valid_action = self.get_valid_action(state)
        if deep == 0:
            for action in range(4):
                if valid_action[action] == 0:
                    next_state, done, reward = self.commands[action](state)
                    father = action
                    value[action] = reward
                    if done:
                        next_state = logic.add_two(next_state, self.difficulty)
                    self.mcts(next_state, value, deep + 1, father)
                else:
                    value[action] = -999
        if deep < 4 and deep != 0:
            for action in range(4):
                if valid_action[action] == 0:
                    next_state, done, reward = self.commands[action](state)
                    if done:
                        next_state = logic.add_two(next_state, self.difficulty)
                        # 开始检查状态
                        if logic.game_state(next_state) == 'lose':
                            return
                    _, empty_cells = logic.check_Is_full(next_state)
                    empty_cells_reward = empty_cells / 2
                    if reward < 0.48:
                        if empty_cells < 5:
                            reward = reward * (16 - empty_cells) / 3
                        else:
                            reward = 0
                    value[father] += (reward + empty_cells_reward) / (4 - np.sum(valid_action))
                    self.mcts(next_state, value, deep + 1, father)

    def take_action(self, action):
        self.matrix, done, _ = self.commands[action](self.matrix)
        if done:
            self.matrix = logic.add_two(self.matrix, self.difficulty)
            self.update_grid_cells()

    def test_search_tree(self):
        self.enable_graph()
        self.master.bind("<Key>", self.mcts_action)

    def test_reinforce(self, use_graph):
        self.agent = PPO(0, 0, 0, 0, 0, 0)
        self.agent.load_weight(327600)
        self.agent.eval()
        self.enable_graph()
        self.master.bind("<Key>", self.reinforce)

    def predict(self, test_case, test_epoch):
        self.test_epoch = test_epoch
        if test_case == "mcts":
            self.test_search_tree()


# 运行
if __name__ == '__main__':
    #0,1,2 难度系数
    difficult_factor=1
    x = GameGrid(difficult_factor)
    # reinforce
    test_case = "mcts"
    x.predict(test_case,5)
