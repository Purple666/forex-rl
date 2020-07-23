import random
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from collections import deque
import tensorflow as tf

from cbam import ChannelGlobalAvgPool1D, ChannelGlobalMaxPool1D
from iqn1 import Agent as iqn_agent
# from memory import Memory
from network import model4 as model
from noisy_dense import IndependentDense

custom_objects = {
    "IndependentDense": IndependentDense,
    "ChannelGlobalMaxPool1D": ChannelGlobalMaxPool1D,
    "ChannelGlobalAvgPool1D": ChannelGlobalAvgPool1D,
}


class Agent(iqn_agent):
    leverage = 300
    ar = 0.05
    money = 1000000
    max_size = 1000000
    gamma = 0.3

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
        # self.x = np.round(self.x, 3)

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

        tau = np.random.uniform(0, 1, (len(actions), 32))
        with tf.GradientTape() as tape:
            q = self.model([states, tau])
            q_backup = q.numpy()

            for i in range(len(actions)):
                q_backup[i, actions[i]] = rewards[i] + self.gamma ** gamma[i] * target_q[i, target_a[i]]

            error = q_backup - q
            tau = tau.reshape((-1, 1, 32))

            huber = tf.where(abs(error) <= 2, error ** 2 * .5, .5 * 2 ** 2 + 2 * tf.abs(error) - 2)
            loss = tf.maximum(tau * huber, (tau - 1) * huber)

            error = tf.reduce_sum(tf.reduce_sum(loss, 1), -1)
            loss = tf.reduce_mean(error)  # * 0.5
            # loss = tf.reduce_mean(error * isw)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        # gradients = [tf.clip_by_value(g, -1, 1) for g in gradients]
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.target_model.set_weights(
            0.001 * np.array(self.model.get_weights()) + 0.999 * np.array(self.target_model.get_weights()))

    def step(self, types=0):
        self.reset += 1
        end = self.step_size - 2
        train = True if types == 0 else False
        step = range(1000) if train else range(50)
        # step = range(10) if train else range(50)
        self.exp = []

        step_size = self.step_size if not train else self.step_size * 2
        for epoch in step:
            # if (1 + self.reset) % 50 == 0:
            #     self.b = np.min((5120, int(self.b * 1.1)))
            # s = np.random.randint(self.y.shape[0])
            s = 0
            if types == 2:
                h = np.random.choice(self.test_step2)
            elif types == 1:
                h = np.random.choice(self.train_step)
            else:
                h = np.random.choice(self.train_step)
            #     # print(h)
            # #     if (1 + epoch) % 1 == 0:
            # #         h += self.step_size
            # #         # h = np.random.choice(self.train_step)
            # # h = self.h

            self.df = df = self.x[s, h:h + step_size]
            self.trend = trend = self.y[s, h:h + step_size]
            atr = self.atr[s, h:h + step_size]

            if not train:
                old_a = 0
                old_a2 = 0
                position = 0
                self.pip = []
                lot = 0
                money = self.money
                b = False

                tau = np.random.uniform(0, 1, (step_size, 32))
                self.a = action = np.argmax(np.mean(self.q([df, tau]), -1), -1)

                for idx, action in zip(range(len(trend)), action):
                    action = 0 if action == 0 else -1 if action == 1 else 1
                    if old_a != action and idx != 0:
                        r = old_a * (trend[idx] - position) - self.spread
                        self.pip.append(r)
                        money += (r * lot)
                        lot = 0
                        if 0 >= money:
                            # b = True
                            money = 0
                            break

                    if (action == -1 or action == 1) and lot == 0:
                        position = trend[idx]
                        lot = money * self.ar / (position / self.leverage)

                    old_a = action

                # self.exp.append(np.sum(self.pip))
                if not b:
                    self.exp.append(((money - self.money) / self.money) * 100)

            else:
                old_a = 0
                old_a2 = 0
                position = 0
                old_idx = 0
                rew = []
                actions = []

                # df = np.random.normal(df, np.abs(df * 0.1))
                # trend = np.random.normal(trend, np.abs(trend * 0.05))

                if self.reset > self.max_size or self.restore is True:

                    tau = np.random.uniform(0, 1, (step_size, 32))
                    q = np.mean(self.q([df, tau]), -1)
                    action = np.argmax(q, -1)
                    action = [a if np.random.randn() > 0.05 else np.random.randint(self.action_size) for a in action]
                else:
                    action = np.random.randint(self.action_size, size=step_size)
                # #
                # for idx, action in zip(range(step_size - self.n), action):
                #     actions.append(action)
                #     action = 0 if action == 0 else -1 if action == 1 else 1
                #
                #     # r = action * (trend[idx + 1] - trend[idx]) - self.spread * np.abs(old_a - action)
                #     # r = np.clip(r, -1, 1)
                #     # rew.append((r / atr[idx]) * 100)
                #     # rew.append(r / 10)
                #     if idx != 0:
                #         r = old_a * (trend[idx] - trend[idx - 1]) - self.spread * np.abs(old_a - old_a2)
                #         # r = np.clip(r, -atr[idx - 1], atr[idx - 1])
                #     else:
                #         r = 0
                #     rew.append(r)
                #
                #     old_a2 = old_a
                #     old_a = action
                #     # e = df[idx], actions[idx], r, df[idx + 1], 0.3
                #     # self.memory.append(e)
                #     # self.reset += 1
                #     # if len(self.memory) >= self.max_size:
                #     #     self.memory.pop(0)
                #     # if self.reset >= self.max_size and len(self.memory) < 500:
                #     #     self.reset = 0
                #
                #     # if len(rew) >= self.n:
                #     #     r = np.sum(rew[-self.n:]) * 0.99 ** self.n
                #     #     e = df[idx - (self.n - 1)], actions[-self.n], r, df[idx + 1], 0.3
                #     #     self.memory.append(e)
                #     #     self.reset += 1
                #     #     if len(self.memory) >= self.max_size:
                #     #         self.memory.pop(0)
                #
                # for idx in range(step_size - self.n):
                #     r = np.sum(rew[idx:idx+self.n - 1] * self.n_)
                #     e = df[idx], actions[idx], r, df[idx+self.n], self.n
                #     self.memory.append(e)
                #     self.reset += 1
                #     if len(self.memory) >= self.max_size:
                #         self.memory.pop(0)
                #         # self.train(None, self.b)
                #         # self.i += 1

                for idx in range(step_size - 1):
                    a = action[idx]
                    a = 0 if a == 0 else -1 if a == 1 else 1
                    if old_a != a:
                        if idx != 0:
                            r = old_a * (trend[idx] - position) - self.spread
                            gamma = idx - old_idx
                            # if r != 0:
                            #     r = (r / np.abs(r)) * (idx - old_idx)
                            # r -= (atr[old_idx] // 2)
                            # r = np.clip(r, -atr[old_idx], atr[old_idx])
                            # r = (r / atr[old_idx]) * 100 #if a != 0 else 0
                            # r = np.clip(r, -1, 1)
                            # e = df[old_idx], action[old_idx], r, df[idx], gamma
                            e = df[idx], action[idx], r, df[idx + 1], 1
                            #             mem.append(r)
                            if len(self.memory) >= self.max_size:
                                self.memory.pop(0)
                            self.reset += 1
                            # if self.reset == int(self.max_size * 2) or (self.reset >= self.max_size and len(self.memory) < 500):
                            #     self.reset = 0
                            #     self.memory = []
                            #     self.restore = True
                            # if self.reset > 100000:
                            #     self.train(e, self.b)
                            self.memory.append(e)
                            # if (self.reset + 1) % 20 == 0:
                            #     self.train()
                        old_idx = idx
                        position = trend[idx]
                    old_a = a
                #
                # # # self.mem = mem
                # # # self.i += 1
                # # # self.train()
                # # # self.reset += 1
                if self.reset > (0.9 * self.max_size):
                    self.train(None, self.b)
                    self.i += 1
        # if train:
        #     self.train(self.b)

    def run(self):
        train_h = []
        test_h = []
        for idx in range(1000000):
            start = time.time()
            # if idx % 10 == 0:
            #     self.h = np.random.choice(self.train_step)
            self.step(0)

            # if True:#self.reset > self.max_size:
            if self.reset > self.max_size:
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

                # self.reset = 25000
                # self.memory = self.memory[0:25000]
