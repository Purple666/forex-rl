import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from rl.agent.dqn import Agent as dqn_Agent
from rl.network.iqn_network2 import model1 as model, custom_objects


class Agent(dqn_Agent):
    memory_size = 1000000
    leverage = 500
    step_size = 240
    replay_ratio = 40
    name = ["iqn.h5", "iqn_e"]
    action_size = 3

    def model_name(self):
        self.custom_objects = custom_objects
        self.build_model = model
        self.input_name = ["i", "t", "p"]
        self.output_name = "q"

    def train(self):
        replay, tree_idx, is_weight = self.memory.sample(self.bach_size)

        states = np.array([a[0] for a in replay], np.float32)
        position_value = np.array([a[1] for a in replay], np.float32).reshape((-1, 1))
        new_states = np.array([a[4] for a in replay], np.float32)
        new_position_value = np.array([a[5] for a in replay], np.float32).reshape((-1, 1))
        actions = np.array([a[2] for a in replay]).reshape((-1,))
        rewards = np.array([a[3] for a in replay], np.float32).reshape((-1,))
        end = np.array([a[6] for a in replay]).reshape((-1,))

        tau = np.random.uniform(0, 1, (len(actions), 32))

        target_q = self.target_q([new_states, tau, new_position_value])
        target_a = np.argmax(np.mean(self.q([new_states, tau, new_position_value]), -1), -1).reshape((-1,))

        with tf.GradientTape() as tape:
            q = self.model([states, tau, position_value])
            q_backup = q.numpy()

            for i in range(len(tree_idx)):
                q_backup[i, actions[i]] = rewards[i] + end[i] * self.gamma ** (self.n - 1) * target_q[i, target_a[i]]

            error = q_backup - q
            error2 = error.numpy()
            error = tf.abs(error)
            loss = 0
            for i in range(self.action_size):
                loss_ = error[:, i, :]
                loss_ = tf.where(loss_ <= 2, loss_ ** 2 * .5, .5 * 2 ** 2 + 2 * (loss_ - 2))
                loss_ = tf.where(error2[:, i, :] < 0, np.abs(tau - 1) * loss_, tau * loss_)
                loss += tf.reduce_mean(tf.reduce_sum(loss_, -1))

        self.error = np.mean(np.mean(np.sum(error, 1), 1))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.target_model.set_weights(
            0.001 * np.array(self.model.get_weights()) + 0.999 * np.array(self.target_model.get_weights()))

        ae = np.mean(np.sum(error, 1), -1).reshape((-1,))
        self.memory.update(tree_idx, ae)

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
            i = np.random.randint(self.y.shape[0], size=3)

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

                tau = np.random.uniform(0, 1, (1, 32))

                position_value = 0
                while True:
                    df_ = df[idx]
                    df_ = np.array([df_])
                    position_value = np.array([[position_value]])
                    q = self.q([df_, tau, position_value])[0]
                    q = np.mean(q, -1)
                    a = np.argmax(q)
                    self.a.append(a)

                    p_idx = idx + 3
                    # if p_idx > (self.step_size - 2):
                    #     a = 0
                    #     self.a[-1] = a

                    if a == 1 and len(position) <= 500:
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
                    position_value = np.sum((trend[p_idx] - np.array(position)) / np.array(lot)) if position else 0

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
        # step_size = self.train_step_size
        step_size = self.range_[-1]
        for i in range(1000000000):
            s = np.random.randint(self.y.shape[0])
            h = 0
            # h = np.random.randint(self.train_step[-1])
            df = self.x[s, h:h+step_size]
            trend = self.y[s, h:h+step_size]

            actions = self.actions

            pip = []
            rew = []
            memory = []
            qv = []

            position = []
            lot = []
            position_value = 0

            old_a = 0
            end = 1

            idx = 0
            old_idx = 0

            while True:
                tau = np.random.uniform(0, 1, (1, 32))
                df_ = df[idx]
                df_ = np.array([df_])
                position_value = np.array([[position_value]])
                q = self.q([df_, tau, position_value])[0]
                q = np.mean(q, -1)
                assert q.shape == (self.action_size,)
                if (position_value[0, 0] >= 20 and np.random.rand() <= 0.1) or (position_value[0, 0] <= -10 and np.random.rand() <= 0.4):
                    a = 0
                elif np.random.rand() >= 0.05:
                    a = np.argmax(q)
                else:
                    a = np.random.randint(self.action_size)
                actions[idx] = a
                qv.append(q[a])

                r = 0
                p_idx = idx + np.random.randint(3, 6)

                if p_idx > (step_size - 1):
                    p_idx = step_size - 1
                    # a = 0

                if a == 1:
                    if len(position) <= 500:
                        position.append(trend[idx])
                        lot.append(position[-1] / self.leverage)
                        if old_idx == 0:
                            old_idx = idx
                    # else:
                    #     r -= 1

                elif a == 0 and position:
                    position = np.array(position)
                    lot = np.array(lot)

                    p = trend[idx] - position
                    r = p / lot
                    r = np.sum(r)
                    # r = 1 if r > 0 else -1
                    # old_idx = (idx - old_idx) * 0.1
                    # r = r / old_idx
                    # end = 0

                    position = []
                    lot = []
                    old_idx = 0

                old_a = a

                rew.append(r)
                e = [df[idx], position_value[0], actions[idx], r, df[p_idx], position_value[0], end]
                memory.append(e)
                idx = p_idx

                if len(rew) >= self.n:
                    e = memory[-self.n]
                    if self.n != 2:
                        r = np.sum(rew[-self.n:-1] * self.n_)
                        e[3] = r
                        e[4] = memory[-1][0]
                        e[5] = memory[-1][1]
                    else:
                        r = e[3]
                        e[5] = memory[-1][1]

                    error = np.abs((r + e[-1] * self.gamma ** (self.n - 1) * qv[-1]) - qv[-self.n])
                    self.memory.add(error, e)
                    self.reset += 1

                end = 1

                if self.reset >= 1000 and self.reset % self.replay_ratio == 0:
                    self.i += 1
                    self.train()
                    errors.append(self.error)

                if self.reset > 1000 and (time.time() - start) >= 120:
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

                # if p_idx >= (step_size - 1):
                #     break
                if p_idx >= self.range_[-1]:
                    break

                # position_value = np.sum((trend[p_idx] - np.array(position)) / np.array(lot)) / ((p_idx - old_idx) * 0.1) if position else 0
                position_value = np.sum((trend[p_idx] - np.array(position)) / np.array(lot)) if position else 0


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
