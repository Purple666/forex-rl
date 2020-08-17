from rl.network.iqn_netowk import model1 as model, custom_objects
from rl.agent.dqn import Agent as dqn_Agent
import numpy as np
import tensorflow as tf
import shutil
import time

import matplotlib.pyplot as plt
import pandas as pd


class Agent(dqn_Agent):
    name = ["iqn.h5", "iqn_e"]

    def model_name(self):
        self.custom_objects = custom_objects
        self.build_model = model
        self.input_name = ["i", "t"]
        self.output_name = "q"

    def train(self):
        replay, tree_idx, is_weight = self.memory.sample(self.bach_size)

        states = np.array([a[0] for a in replay], np.float32)
        new_states = np.array([a[3] for a in replay], np.float32)
        actions = np.array([a[1] for a in replay]).reshape((-1,))
        rewards = np.array([a[2] for a in replay], np.float32).reshape((-1,))
        end = np.array([a[4] for a in replay]).reshape((-1,))

        tau = np.random.uniform(0, 1, (len(actions), 32))
        target_q = self.target_q([new_states, tau])
        target_a = np.argmax( np.mean(self.q([new_states, tau]), -1), -1 ).reshape((-1,))

        with tf.GradientTape() as tape:
            q = self.model([states, tau])
            q_backup = q.numpy()

            for i in range(len(tree_idx)):
                q_backup[i, actions[i]] = rewards[i] + end[i] * self.gamma ** (self.n - 1) * target_q[i, target_a[i]]

            error = q_backup - q
            error = tf.reduce_sum(error, 1)
            tau[error < 0] -= 1
            tau = np.abs(tau)
            error = tf.abs(error)
            loss = tf.where(error <= 1, error ** 2 * .5, .5 * 1 ** 2 + 1 * (error - 1))
            loss = tau * loss
            loss = tf.reduce_mean(loss, -1)
            loss = tf.reduce_mean(loss * is_weight)

        self.error = loss
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.target_model.set_weights(
            0.001 * np.array(self.model.get_weights()) + 0.999 * np.array(self.target_model.get_weights()))

        ae = np.mean(error.numpy(), -1).reshape((-1,))
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

                tau = np.random.uniform(0, 1, (self.step_size, 32))
                self.a = action = np.argmax(np.mean(self.q([df, tau]), -1), -1)

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
            qv = []

            for idx in self.range_:
                tau = np.random.uniform(0, 1, (1, 32))
                df_ = np.array([df[idx]])
                q = self.q([df_, tau])[0]
                q = np.mean(q, -1)
                assert q.shape == (self.action_size,)
                if actions[idx] is None:
                    if np.random.rand() <= 0.25:
                        a = actions[idx - 1]
                    elif np.random.rand() >= 0.1:
                        a = np.argmax(q)
                    else:
                        a = np.random.randint(self.action_size)
                    actions[idx] = a
                else:
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
