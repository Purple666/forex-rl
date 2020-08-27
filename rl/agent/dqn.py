import shutil
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from rl.memory.memory import Memory
from rl.network.dqn_network import model1 as model, custom_objects
import tensorflow_addons as tfa


class Agent:
    memory_size = 10000
    step_size = 500
    money = 10000000
    leverage = 500
    ar = 0.05
    spread = 10
    name = ["dqn.h5", "dqn_e"]
    bach_size = 64
    replay_ratio = 40  # 0.25
    reset = 0
    action_size = 3

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
        self.memory = Memory(self.memory_size)
        self.actions = [None for _ in range(self.train_step[-1])]
        self.range_ = range(self.train_step[-1] - 3)

    def env(self):
        t = 1
        x = np.load(f"rl/data/x{t}.npy")
        shape = x.shape
        self.x = x.reshape((shape[0], -1, shape[-2], shape[-1]))
        y = np.load(f"rl/data/target{t}.npy")
        shape = y.shape
        y = y.reshape((shape[0], y.shape[2], -1))
        self.y, self.v, self.atr, self.high, self.low = \
            y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4]

        self.train_step = np.arange(0, int(self.x.shape[1] - self.x.shape[1] * 0.2 - self.step_size),
                                    self.step_size)
        # self.train_step = np.arange(0, int(self.x.shape[1] - self.x.shape[1] * 0.2 - self.step_size))
        self.test_step = self.train_step[-1] + self.step_size, self.x.shape[1] - self.step_size
        self.test_step2 = np.arange(self.test_step[0], self.test_step[1], self.step_size)

    def model_name(self):
        self.custom_objects = custom_objects
        self.build_model = model
        self.input_name = "i"
        self.output_name = "q"

    def build(self):
        print(self.x.shape[-2:])
        if self.restore:
            self.model = tf.keras.models.load_model(self.restore_path + self.name[0], custom_objects=self.custom_objects)
            self.i = np.load(self.restore_path + self.name[1] + ".npy")
        else:
            self.model = self.build_model(self.x.shape[-2:], self.action_size, self.dueling, self.noisy)
            opt = tfa.optimizers.Lookahead(tf.keras.optimizers.Nadam(self.lr))
            self.model.compile(opt)
            self.i = 0
        self.target_model = self.build_model(self.x.shape[-2:], self.action_size, self.dueling, self.noisy)
        self.target_model.set_weights(self.model.get_weights())

        get = self.model.get_layer
        inputs = get(self.input_name).input if type(self.input_name) == str else [get(i).input for i in self.input_name]
        self.q = tf.keras.backend.function(inputs, get(self.output_name).output)
        get = self.target_model.get_layer
        inputs = get(self.input_name).input if type(self.input_name) == str else [get(i).input for i in self.input_name]
        self.target_q = tf.keras.backend.function(inputs, get(self.output_name).output)

    def train(self):
        # replay = random.sample(self.memory, self.bach_size)
        replay, tree_idx, is_weight = self.memory.sample(self.bach_size)

        states = np.array([a[0] for a in replay], np.float32)
        new_states = np.array([a[3] for a in replay], np.float32)
        actions = np.array([a[1] for a in replay]).reshape((-1,))
        rewards = np.array([a[2] for a in replay], np.float32).reshape((-1,))
        end = np.array([a[4] for a in replay]).reshape((-1,))

        target_q = self.target_q(new_states)
        target_a = np.argmax(self.q(new_states), -1)

        with tf.GradientTape() as tape:
            q = self.model(states)
            q_backup = q.numpy()

            for i in range(len(end)):
                q_backup[i, actions[i]] = rewards[i] + end[i] * self.gamma ** (self.n - 1) * target_q[i, target_a[i]]

            error = tf.abs(q_backup - q)
            # loss = tf.where(error <= 2, error ** 2 * .5, .5 * 2 ** 2 + 2 * (error - 2)) * is_weight
            # loss = tf.reduce_mean(loss)
            loss = 0
            for i in range(self.action_size):
                loss += tf.reduce_mean(error[:, i] ** 2) * 0.5
            # loss = tf.reduce_mean(error[:,0] ** 2 * is_weight) * .5

        self.error = np.mean(np.sum(np.abs(error), -1))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [tf.clip_by_value(gradients, -1, 1) for gradients in gradients]
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # if (self.i + 1) % 4000 == 0:
        #     self.target_model.set_weights(self.model.get_weights())

        self.target_model.set_weights(
            0.001 * np.array(self.model.get_weights()) + 0.999 * np.array(self.target_model.get_weights()))

        ae = np.sum(error, -1)
        self.memory.update(tree_idx, ae)

    def step(self, types=1):
        self.exp = []
        h = 0
        for _ in range(5):
            if types == 2:
                h = np.random.choice(self.test_step2)
            elif types == 1:
                h = np.random.choice(self.train_step)
            total_money = self.money
            self.pip = []
            self.roi = []
            self.pip_ = []

            for i in range(self.y.shape[0]):
                self.df = df = self.x[i, h:h + self.step_size]
                trend = self.y[i, h:h + self.step_size]
                atr = self.atr[i, h:h + self.step_size]

                position = 0
                lot = 0
                self.a = action = np.argmax(self.q(df), -1)
                idx = 0
                old_a = 0

                while True:
                    a = action[idx]
                    a = 0 if a == 0 else -1 if a == 1 else 1

                    if old_a != a:
                        if old_a != 0:
                            pip = old_a * (trend[idx] - position) - self.spread * np.abs(a)
                            self.pip.append(pip)
                            roi = pip / lot
                            self.roi.append(roi)
                        position = trend[idx]
                        lot = trend[idx] / self.leverage

                    old_a = a
                    idx += 5

                    if idx >= (self.step_size - 5):
                        break

            roi = np.mean(self.roi) * 100 if self.roi else 0
            self.exp.append(roi)

    def run(self):
        train_h = []
        test_h = []
        seed = 0
        errors = []
        start = time.time()
        s = 0
        for i in range(10000000):
            if i % 10 == 0:
                s = np.random.randint(self.y.shape[0])
                df = self.x[s]
                atr = self.atr[s]
                trend = self.y[s]
                matr = np.max(atr)
            money = self.money
            old_money = money
            a = 0
            end = 1
            old_a = None
            old_idx = 0
            self.mem = memory = []
            self.rew = rew = []
            actions = self.actions[:]
            qv = []
            self.qv = qv_ = []
            idx = 0
            p_idx = 0
            lot = []
            position = []
            losscut = False

            while end == 1:
                df_ = df[idx]
                df_ = np.array([df_])
                q = self.q([df_])[0]
                if np.random.rand() >= 0.05:
                    a = np.argmax(q)
                else:
                    a = np.random.randint(self.action_size)
                actions[idx] = a
                qv.append(q[a])
                a = 0 if a == 0 else -1 if a == 1 else 1
                pidx = idx + np.random.randint(3, 10)

                pip = a * (trend[pidx] - trend[idx]) - self.spread * np.abs(a)
                lot = trend[idx] / self.leverage
                r = pip / lot

                rew.append(r)
                e = [df[idx], actions[idx], r, df[pidx], 1]
                memory.append(e)
                idx = pidx

                if len(rew) >= self.n:
                    e = memory[-self.n]
                    if self.n != 2:
                        r = np.sum(rew[-self.n:-1] * self.n_)
                        e[2] = r
                        e[3] = memory[-1][3]
                    else:
                        r = e[2]

                    error = np.abs((r + self.gamma ** (self.n - 1) * qv[-1]) - qv[-self.n])
                    self.memory.add(error, e)
                    self.reset += 1

                if self.reset >= 1000 and self.reset % self.replay_ratio == 0:
                    self.i += 1
                    self.train()
                    if self.i > 10:
                        errors.append(self.error)

                if self.reset > 1000 and (time.time() - start) >= 60:
                    if i % 20 == 0:
                        seed = self.i
                    np.random.seed(seed)

                    train = []
                    test = []
                    lev = self.leverage
                    # self.leverage = 200
                    for _ in range(1):
                        self.step(1)
                        train.extend(self.exp)
                        self.step(2)
                        test.extend(self.exp)
                    self.leverage = lev

                    # self.model.set_weights(w)

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

                if pidx >= self.range_[-2]:
                    break


if __name__ == "__main__":
    # "lr=1e-3, action_size=3, dueling=True, noisy=True, n=3, restore=False, restore_path="rl/save_mode"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--action_size", type=int, default=3)
    parser.add_argument("--dueling", type=bool, default=True)
    parser.add_argument("--noisy", type=bool, default=True)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--restore", type=bool, default=False)
    parser.add_argument("--restore_path", type=str, default="rl/save_mode/")
    args = parser.parse_args()

    agent = Agent(args.lr, args.action_size, args.dueling, args.noisy, args.n, args.restore, args.restore_path)
    agent.run()
