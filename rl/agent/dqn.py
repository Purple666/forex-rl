import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from rl.memory.memory import Memory
from rl.network.dqn_network import model, custom_objects


class Agent:
    memory_size = 10000000
    step_size = 500
    money = 1000000
    leverage = 500
    ar = 0.05
    spread = 10
    name = ["dqn.h5", "dqn_e"]
    gamma = 0.99
    bach_size = 64
    replay_ratio = 4  # 0.25
    reset = 0

    def __init__(self, lr=1e-3, action_size=3, dueling=True, noisy=True, n=3, restore=False, restore_path="rl/save_model/"):
        self.lr = lr
        self.action_size = action_size
        self.dueling = dueling
        self.noisy = noisy
        self.n = n + 1
        self.n_ = np.array([self.gamma ** i for i in range(self.n - 1)])
        self.restore = restore
        self.restore_path = restore_path
        self.model_name()
        self.env()
        self.build()
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
            self.model.compile(tf.keras.optimizers.Adam(self.lr))
            self.i = 0
        self.target_model = self.build_model(self.x.shape[-2:], self.action_size, self.dueling, self.noisy)
        self.target_model.set_weights(self.model.get_weights())

        get = self.model.get_layer
        self.q = tf.keras.backend.function(get(self.input_name).input, get(self.output_name).output)
        get = self.target_model.get_layer
        self.target_q = tf.keras.backend.function(get(self.input_name).input, get(self.output_name).output)

    def train(self):
        tree_idx, replay = self.memory.sample(self.bach_size)

        states = np.array([a[0][0] for a in replay], np.float32)
        new_states = np.array([a[0][3] for a in replay], np.float32)
        actions = np.array([a[0][1] for a in replay]).reshape((-1,))
        rewards = np.array([a[0][2] for a in replay], np.float32).reshape((-1,))
        end = np.array([a[0][4] for a in replay]).reshape((-1,))

        target_q = self.target_q(new_states)
        target_a = np.argmax(self.q(new_states), -1)

        with tf.GradientTape() as tape:
            q = self.model(states)
            q_backup = q.numpy()

            for i in range(len(tree_idx)):
                q_backup[i, actions[i]] = rewards[i] + end[i] * self.gamma ** (self.n - 1) * target_q[i, target_a[i]]

            error = tf.reduce_sum(q_backup - q, -1)
            loss = tf.reduce_mean(error ** 2)

        self.error = loss
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        ae = np.abs(error.numpy().reshape((-1,)))
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

            for i in range(self.y.shape[0]):
                self.df = df = self.x[i, h:h + self.step_size]
                trend = self.y[i, h:h + self.step_size]
                atr = self.atr[i, h:h + self.step_size]

                old_a = 0
                position = 0
                self.pip = []
                self.pip_ = []
                lot = 0
                money = self.money

                self.a = action = np.argmax(self.q(df), -1)

                for idx in range(trend.shape[0] - 3):
                    a = action[idx]
                    a = 0 if a == 0 else -1 if a == 1 else 1
                    if old_a != a:
                        if a != 0:
                            for i_ in range(1, 3):
                                action[idx + i_] = action[idx]
                        if idx != 0:
                            r = old_a * (trend[idx] - position) - self.spread * np.abs(old_a)
                            self.pip_.append(r)
                            r *= lot
                            money += r

                            if money <= 0:
                                money = 0
                                break
                        old_idx = idx
                        position = trend[idx]
                        lot = money * self.ar / (position / self.leverage)
                        # risk = money * self.ar
                    old_a = a
                total_money += (money - self.money)
            self.exp.append(((total_money - self.money) / self.money) + 1)
            self.total_money = total_money

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
            money = self.money
            old_money = money
            a = 0
            end = 1
            old_a = 0
            memory = []
            rew = []
            actions = self.actions[:]
            df = self.x[s]
            atr = self.atr[s]
            trend = self.y[s]
            self.qv = qv = []

            for idx in self.range_:
                df_ = np.array([df[idx]])
                q = self.q(df_)[0]
                if actions[idx] is None:
                    if np.random.rand() <= 0.25:
                        a = actions[idx - 1]
                    elif np.random.rand() >= 0.1:
                        a = np.argmax(q)
                    else:
                        a = np.random.randint(self.action_size)
                    actions[idx] = a
                    a = 0 if a == 0 else -1 if a == 1 else 1
                qv.append(q[a])

                if old_a != a and a != 0:
                    for i_ in range(1, 3):
                        actions[idx + i_] = actions[idx]

                lot = money * self.ar / (trend[idx - 1] / self.leverage)
                r = (a * (trend[idx + 1] - trend[idx]) - self.spread * np.abs(a)) * lot
                money += r
                r = ((money - old_money) / old_money) * 10

                if money <= 0:
                    end = 0
                    r = ((money - self.money) / self.money) * 10

                e = [df[idx], actions[idx], r, df[idx + 1], end]
                rew.append(r)
                memory.append(e)

                if len(rew) >= self.n:
                    e = memory[-self.n]
                    r = np.sum(rew[-self.n:-1] * self.n_) if end == 1 else r

                    error = np.abs((r + end * self.gamma ** (self.n - 1) * qv[-1]) - qv[-self.n])

                    e[2] = r
                    e[3] = memory[-1][0]
                    if end == 0:
                        e[4] = 0
                    self.memory.add(error, e)
                    self.reset += 1

                if self.reset >= 1000 and self.reset % self.replay_ratio == 0:
                    self.i += 1
                    self.train()
                    errors.append(self.error)

                old_a = a
                old_money = money

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

                if end == 0:
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
