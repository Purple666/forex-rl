import random
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from collections import deque
import tensorflow as tf
# import copy

from cbam import ChannelGlobalAvgPool1D, ChannelGlobalMaxPool1D
from iqn1 import Agent as iqn_agent
# from memory import Memory
from network import model2 as model
from noisy_dense import IndependentDense

custom_objects = {
    "IndependentDense": IndependentDense,
    "ChannelGlobalMaxPool1D": ChannelGlobalMaxPool1D,
    "ChannelGlobalAvgPool1D": ChannelGlobalAvgPool1D,
}


class Agent(iqn_agent):
    leverage = 500
    ar = 0.1
    money = 1000000
    # max_size = 0
    max_size = 1000000
    # gamma = 0.67
    gamma = 0.85

    def __init__(self, action_size=3, lr=1e-3, spread=5, step_size=200, n=3, restore=False):
        self.spread = spread
        self.action_size = action_size
        self.step_size = step_size
        self.lr = lr
        self.n = n
        self.n_ = np.array([self.gamma ** i for i in range(self.n - 1)])
        self.restore = restore
        self.memory = []
        self.state()
        self.build_model = model
        self.build()
        self.reset = 0
        self.b = 64

    def train(self, exp=None, b=128):
        replay = random.sample(self.memory, b)
        # replay = np.random.randint(0, len(self.memory), b)
        # replay = np.sort(replay)[::-1]
        # replay = [self.memory.pop(i) for i in replay]
        if exp is not None:
            replay += [exp]

        self.states = states = np.array([a[0] for a in replay], np.float32)
        new_states = np.array([a[3] for a in replay], np.float32)
        actions = np.array([a[1] for a in replay]).reshape((-1, 1))
        rewards = np.array([a[2] for a in replay], np.float32).reshape((-1, 1))
        gamma = np.array([a[4] for a in replay]).reshape((-1, 1))

        self.tau = tau = np.random.uniform(0, 1, (len(actions), 32))
        target_tau = np.random.uniform(0, 1, (len(actions), 32))

        target_q = self.target_q([new_states, target_tau])
        target_a = np.argmax(np.sum(self.q([new_states, tau]), -1), -1)

        with tf.GradientTape() as tape:
            q = self.model([states, tau])
            q_backup = q.numpy()

            for i in range(len(actions)):
                q_backup[i, actions[i]] = rewards[i] + self.gamma ** gamma[i] * target_q[i, target_a[i]]


        # target_q = self.target_q([new_states, target_tau])
        # target_q = np.mean(target_q, -1)
        # target_a = np.argmax(np.sum(self.q([new_states, tau]), -1), -1)
        #
        # tau = np.random.uniform(0, 1, (len(actions), 32))
        # with tf.GradientTape() as tape:
        #     q = self.model([states, tau])
        #     q_backup = q.numpy()
        #
        #     for i in range(len(actions)):
        #         q_backup[i, actions[i]] = np.tile(rewards[i] + (self.gamma ** gamma[i]) * target_q[i, target_a[i]], 32)

            error = q_backup - q
            tau = tau.reshape((-1, 1, 32))

            huber = tf.where(abs(error) <= 2, error ** 2 * .5, .5 * 2 ** 2 + 2 * (tf.abs(error) - 2))
            loss = tf.maximum(tau * huber, (tau - 1) * huber)

            error = tf.reduce_sum(tf.reduce_sum(loss, 1), -1)
            loss = tf.reduce_mean(error)  # * 0.5
            # loss = tf.reduce_mean(error * isw)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [tf.clip_by_value(g, -100, 100) for g in gradients]
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.target_model.set_weights(
            0.001 * np.array(self.model.get_weights()) + 0.999 * np.array(self.target_model.get_weights()))

    def step(self, types=0):
        end = self.step_size - 2
        train = True if types == 0 else False
        step = range(100) if train else range(50)
        # step = range(10) if train else range(50)
        self.exp = []

        step_size = self.step_size if not train else self.step_size * 5
        h = np.random.choice(self.train_step)
        for epoch in step:
            # if (1 + self.reset) % 50 == 0:
            #     self.b = np.min((5120, int(self.b * 1.1)))
            s = np.random.randint(self.y.shape[0])
            # s = 0
            if types == 2:
                h = np.random.choice(self.test_step2)
            elif types == 1:
                h = np.random.choice(self.train_step)
            else:
                # h = np.random.choice(self.train_step)
                h += self.step_size
                if h > self.train_step[-5]:
                    h = np.random.choice(self.train_step)

            self.df = df = self.x[s, h:h + step_size]
            self.trend = trend = self.y[s, h:h + step_size]
            atr = self.atr[s, h:h + step_size]

            if not train:
                old_a = 0
                old_a2 = 0
                position = 0
                self.pip = []
                self.pip_ = []
                lot = 0
                money = self.money
                b = False
                old_idx = 0

                tau = np.random.uniform(0, 1, (step_size, 32))
                self.a = action = np.argmax(np.mean(self.q([df, tau]), -1), -1)

                for idx in range(step_size - 3):
                    a = action[idx]
                    a = 0 if a == 0 else -1 if a == 1 else 1
                    if old_a != a:
                        if a != 0:
                            for i_ in range(1, 3):
                                action[idx + i_] = action[idx]
                        if idx != 0:
                            r = old_a * (trend[idx] - position) - self.spread * np.abs(old_a)
                            self.pip_.append(r)
                            r = np.clip(r, -atr[old_idx], atr[old_idx] * 2) * lot
                            self.pip.append(r)
                            money += r
                        old_idx = idx
                        position = trend[idx]
                        lot = money * self.ar / (position / self.leverage)
                    old_a = a

                # self.exp.append(np.sum(self.pip))
                if not b:
                    self.exp.append(((money - self.money) / self.money) * 100)

            else:
                old_a = 0
                old_a2 = 0
                position = 0
                lot = 0
                old_idx = 0
                rew = []
                actions = []
                mem = []

                money = self.money
                old_money = money
                # df = np.random.normal(df, np.abs(df * 0.1))
                # trend = np.random.normal(trend, np.abs(trend * 0.05))

                tau = np.random.uniform(0, 1, (step_size, 32))
                q = np.mean(self.q([df, tau]), -1)
                action = np.argmax(q, -1)
                random_a = np.random.randint(self.action_size, size=(step_size))
                epsilon = 0.05 if self.restore is True else 0.3
                action = [a if np.random.rand() > epsilon else ra for a, ra in zip(action, random_a)]

                for idx in range(step_size - 5):
                    a = action[idx]
                    a = 0 if a == 0 else -1 if a == 1 else 1
                    if old_a != a:
                        if a != 0:
                            for i_ in range(1, np.random.randint(3, 6)):
                            # for i_ in range(1, 3):
                                action[idx + i_] = action[idx]
                        if idx != 0:
                            r = old_a * (trend[idx] - position) - self.spread * np.abs(old_a)
                            r = np.clip(r, -atr[old_idx], atr[old_idx] * 2) * lot
                            money += r
                            r = ((money - old_money) / old_money)# * 100
                            if money < 0:
                                break
                            e = [df[idx], action[idx], r, df[idx + 1], 1]
                            rew.append(r)
                            mem.append(e)
                            # self.reset += 1
                            # self.memory.append(e)
                        old_idx = idx
                        position = trend[idx]
                        lot = money * self.ar / (position / self.leverage)
                        old_money = money
                    old_a = a

                for idx in range(len(mem) - self.n):
                    try:
                        r = np.sum(rew[idx:idx+self.n-1] * self.n_)
                        mem[idx] = mem[idx][0], mem[idx][1], r, mem[idx+self.n-1][0], self.n - 1
                    except:
                        mem = mem[:idx-1]
                        break
                self.memory.extend(mem)
                self.reset += (len(mem))
                self.a_ = action
                if len(self.memory) > self.max_size:
                    self.memory = self.memory[-self.max_size:]

                if self.reset > (2 * self.b):
                    self.train(None, self.b)
                    self.i += 1

    def run(self):
        train_h = []
        test_h = []
        for idx in range(1000000):
            start = time.time()
            # if idx % 10 == 0:
            #     self.h = np.random.choice(self.train_step)
            self.step(0)
            # if len(self.memory) > (self.max_size * 0.9):
            #     for i in range(100):
            #         self.train(None, 512)
            # self.i += 100

            if self.reset > (2 * self.b):
                train = []
                test = []
                for _ in range(1):
                    self.step(1)
                    train.extend(self.exp)
                    self.step(2)
                    test.extend(self.exp)

                print(f"epoch {self.i}")
                print(f"speed {time.time() - start}sec")
                plt.cla()
                train_h.append(np.median(train))
                test_h.append(np.median(test))

                plt.plot(train_h, label="train")
                plt.plot(test_h, label="test")
                plt.show()

                df = pd.DataFrame({"train": np.array(train),
                                   "test": np.array(test)})
                print(df.describe())

                np.save(self.name[0], self.i)
                self.model.save(self.name[1])

                try:
                    _ = shutil.copy(f"/content/{self.name[1]}", "/content/drive/My Drive")
                    _ = shutil.copy(f"/content/{self.name[0]}.npy", "/content/drive/My Drive")
                except:
                    pass

                # self.memory = self.memory[25000:]
                # self.reset = len(self.memory)
            # self.old_model.set_weights(self.model.get_weights())
