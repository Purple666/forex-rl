import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from rl.agent.dqn import Agent as dqn_Agent
from rl.network.dqn_network import model1 as model, custom_objects


class Agent(dqn_Agent):
    name = ["ebu.h5", "ebu_e"]
    action_size = 3
    beta = 1
    memory_size = 1000
    replay_ratio = 4

    def __init__(self, lr=1e-3, gamma=0.99, dueling=True, noisy=True, n=3, restore=False, restore_path="rl/save_model/"):
        print("1")
        self.lr = lr
        self.dueling = dueling
        self.noisy = noisy
        self.gamma = gamma
        self.n = n if n != 1 else 2
        self.n_ = np.array([self.gamma ** i for i in range(self.n - 1)])
        self.restore = restore
        self.restore_path = restore_path
        self.model_name()
        self.env()
        self.build()
        # self.memory = []
        self.memory = []
        self.actions = np.array([None for _ in range(self.train_step[-1])])
        self.range_ = range(self.train_step[-1])

    def model_name(self):
        self.custom_objects = custom_objects
        self.build_model = model
        self.input_name = ["i", "p"]
        self.output_name = "q"

    def train(self):
        replay = self.memory[-1]

        states = np.array([a[0] for a in replay], np.float32)
        position_value = np.array([a[1] for a in replay], np.float32).reshape((-1, 1))
        new_states = np.array([a[4] for a in replay], np.float32)
        new_position_value = np.array([a[5] for a in replay], np.float32).reshape((-1, 1))
        actions = np.array([a[2] for a in replay]).reshape((-1,))
        rewards = np.array([a[3] for a in replay], np.float32).reshape((-1,))
        end = np.array([a[6] for a in replay]).reshape((-1,))

        target_q = self.target_q([new_states, new_position_value])
        # target_a = np.argmax(self.q([new_states, new_position_value]), -1)

        y = np.zeros((len(actions), 1))
        y[-1] += rewards[-1]
        k = np.arange(len(actions) - 2, -1, -1)
        for k in k:
            target_q[k, actions[k+1]] = self.beta * y[k + 1] + target_q[k, actions[k + 1]] * (1 - self.beta)
            # y[k] = rewards[k] + end[k] * np.max(target_q[k])
            y[k] = rewards[k] + end[k] * target_q[k, actions[k + 1]]

        with tf.GradientTape() as tape:
            q = self.model([states, position_value])
            q_backup = q.numpy()

            for i in range(len(actions)):
                q_backup[i, actions[i]] = y[i]

            error = tf.abs(q_backup - q)
            loss = 0
            for i in range(self.action_size):
                loss += tf.reduce_mean(error[:, i] ** 2) * 0.5
            # loss = tf.reduce_mean(error[:,0] ** 2 * is_weight) * .5

        self.error = np.mean(np.sum(np.abs(error), -1))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # gradients = [tf.clip_by_value(gradients, -1, 1) for gradients in gradients]
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.target_model.set_weights(
            0.001 * np.array(self.model.get_weights()) + 0.999 * np.array(self.target_model.get_weights()))

    def step(self, types=1):
        self.exp = []
        h = 0
        for _ in range(5):
            if types == 2:
                h = np.random.choice(self.test_step2)
            elif types == 1:
                h = np.random.choice(self.train_step)
            self.pip = []
            self.roi = []
            self.a = []
            i = np.random.choice(self.y.shape[0], size=3, replace=False)

            for i in i:
                # for i in range(self.y.shape[0]):
                self.df = df = self.x[i, h:h + self.step_size]
                trend = self.y[i, h:h + self.step_size]
                atr = self.atr[i, h:h + self.step_size]

                idx = 0
                old_idx = 0
                old_a = 0
                position = []
                lot = []

                position_value = 0
                while True:
                    df_ = df[idx]
                    df_ = np.array([df_])
                    position_value = np.array([[position_value]])
                    q = self.q([df_, position_value])[0]
                    a = np.argmax(q)
                    self.a.append(a)

                    p_idx = idx + 3
                    # if p_idx > (self.step_size - 2):
                    #     a = 0
                    #     self.a[-1] = a

                    if a == 1 and len(position) <= 5:
                        position.append(trend[idx])
                        lot.append(position[-1] / self.leverage)

                        if old_idx == 0:
                            old_idx = idx

                    elif a == 0 and position:
                        position = np.array(position)
                        lot = np.array(lot)

                        p = trend[idx] - position
                        self.pip.extend(p)
                        r = p / lot
                        self.roi.extend(r)

                        position = []
                        lot = []
                        old_idx = 0

                    old_a = a
                    idx = p_idx

                    if p_idx > (self.step_size - 2):
                        break

                    # position_value = np.sum((trend[p_idx] - np.array(position)) / np.array(lot)) / (
                    #             (p_idx - old_idx) * 0.1) if position else 0
                    # old_idx_ = ((idx - old_idx) / 5) * 0.1
                    position_value = np.sum(
                        (trend[p_idx] - np.array(position)) / np.array(lot)) / 1 if position else 0

            # self.pip = np.array(self.pip)
            roi = np.sum(self.roi) if self.roi else 0
            self.exp.append(roi)

    def run(self):
        train_h = []
        test_h = []
        seed = 0
        errors = []
        start = time.time()
        s = 0
        step_size = 500
        # step_size = self.range_[-1]
        for i in range(1000000000):
            s = np.random.randint(self.y.shape[0])
            # h = 0
            while True:
                h = np.random.randint(self.train_step[-1])
                if (h + step_size) < self.test_step[0]:
                    break
            # h = np.random.randint(self.train_step[-1])
            df = self.x[s, h:h+step_size]
            trend = self.y[s, h:h+step_size]

            actions = self.actions

            pip = []
            rew = []
            memory = []
            ends = []
            qv = []

            position = []
            lot = []
            position_value = 0

            old_a = 0
            end = 1

            idx = 0
            old_idx = 0

            while True:
                df_ = df[idx]
                df_ = np.array([df_])
                position_value = np.array([[position_value]])
                q = self.q([df_, position_value])[0]
                assert q.shape == (self.action_size,)
                if np.random.rand() >= 0.05:
                    a = np.argmax(q)
                else:
                    a = np.random.randint(self.action_size)
                # if position_value[0, 0] <= -10:
                #     a = 0

                actions[idx] = a
                qv.append(q[a])

                r, r_ = 0, 0
                p_idx = idx + np.random.randint(3, 6)

                if p_idx > (step_size - 1):
                    p_idx = step_size - 1
                    # end = 0

                if a == 1:
                    if len(position) <= 5:
                        position.append(trend[idx])
                        lot.append(position[-1] / self.leverage)
                        if old_idx == 0:
                            old_idx = idx
                    else:
                        r_ -= 1
                    # r = ((trend[p_idx] - trend[idx]) / lot[-1]) * len(position) if position else 0

                elif a == 0 and position:
                    position = np.array(position)
                    lot = np.array(lot)

                    p = trend[idx] - position - self.spread
                    r = p / lot
                    r = np.sum(r)

                    # r_ = np.clip(r, -10, 1000)
                    end = 0 if r > 0 else 1

                    position = []
                    lot = []
                    old_idx = 0

                if p_idx == (step_size - 1) and position:
                    position = np.array(position)
                    lot = np.array(lot)

                    p = trend[idx] - position - self.spread
                    r = p / lot
                    r = np.sum(r)

                    end = 0

                ends.append(end)
                rew.append(r)
                # r = np.sum(rew) if p_idx == (step_size - 1) else r_
                e = [df[idx], position_value[0], actions[idx], r, df[p_idx], position_value[0], end]
                memory.append(e)
                idx = p_idx

                end = 1

                if self.reset >= 1 and (time.time() - start) >= 120:
                    if i % 20 == 0:
                        seed = self.i
                    np.random.seed(seed)

                    train = []
                    test = []
                    for _ in range(1):
                        self.step(1)
                        train.extend(self.exp)
                        self.step(2)
                        test.extend(self.exp)

                    print(f"epoch {self.i}")
                    print(f"speed {time.time() - start}sec")
                    print(f"loss = {self.error}")
                    plt.cla()
                    train_h.append(np.median(train))
                    test_h.append(np.median(test))

                    plt.plot(train_h, label="train")
                    plt.plot(test_h, label="test")
                    plt.show()
                    plt.cla()
                    plt.plot(errors)
                    plt.show()

                    describe = pd.DataFrame({"train": np.array(train),
                                             "test": np.array(test)})
                    print(describe.describe())

                    np.save(self.name[1], self.i)
                    self.model.save(self.name[0])

                    try:
                        _ = shutil.copy(f"/content/{self.name[0]}", "/content/drive/My Drive/fxrl/rl/save_model")
                        _ = shutil.copy(f"/content/{self.name[1]}.npy", "/content/drive/My Drive/fxrl/rl/save_model")
                    except:
                        pass

                    start = time.time()
                    np.random.seed(None)

                if p_idx >= (step_size - 1):
                    break

                # position_value = np.sum((trend[p_idx] - np.array(position)) / np.array(lot)) / ((p_idx - old_idx) * 0.1) if position else 0
                # old_idx_ = ((idx - old_idx) / 5) * 0.1
                position_value = np.sum((trend[p_idx] - np.array(position)) / np.array(lot)) / 1 if position else 0

            self.memory.append(memory)
            if self.reset > self.memory_size:
                self.memory.pop(0)
            self.reset += 1

            end = 1

            if self.reset >= 1:
                self.i += 1
                self.train()
                errors.append(self.error)
