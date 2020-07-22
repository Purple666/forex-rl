import tensorflow as tf
import numpy as np
from network import model1 as model
from copy import copy
from memory import Memory
import time
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from noisy_dense import IndependentDense
from cbam import  ChannelGlobalAvgPool1D, ChannelGlobalMaxPool1D
import tensorflow_addons as tfa


custom_objects={
                "IndependentDense": IndependentDense,
                "ChannelGlobalMaxPool1D": ChannelGlobalMaxPool1D,
                "ChannelGlobalAvgPool1D": ChannelGlobalAvgPool1D}


class Agent:

    sigma = 0.2
    alpha = 1.01
    epsilon = 0.5
    min_epsilon = 0.01
    name = ["iqn_e", "iqn.h5"]
    custom_objects = custom_objects

    def __init__(self, action_size=3, lr=1e-3, n=3, spread=5, step_size=1000, money=10000, leverage=500, restore=False):
        self.n = n
        self.spread = spread
        self.action_size = action_size
        self.step_size = step_size
        self.lr = lr
        self.money = money
        self.leverage = leverage
        self.restore = restore
        self.build_model = model
        self.memory = Memory(50000)
        self.state()
        self.build()
        self.w = self.model.get_weights()
        self.reset = 0
        self.e = []

    def build(self):
        if self.restore:
            self.i = np.load(f"{self.name[0]}.npy")
            self.model = tf.keras.models.load_model(self.name[1], custom_objects=self.custom_objects)
        else:
            self.i = 0
            self.model = self.build_model(self.x.shape[-2:], self.action_size)
            opt = tfa.optimizers.Lookahead(tf.keras.optimizers.Nadam(self.lr))
            # opt =
            self.model.compile(opt)

        self.target_model = self.build_model(self.x.shape[-2:], self.action_size)
        self.target_model.set_weights(self.model.get_weights())

        get = self.model.get_layer
        self.q = tf.keras.backend.function([get("i").input,get("t").input], get("q").output)
        get = self.target_model.get_layer
        self.target_q = tf.keras.backend.function([get("i").input,get("t").input], get("q").output)

    def state(self):
        t = 1
        x = np.load(f"x{t}.npy")
        shape = x.shape
        self.x = x.reshape((shape[0], -1, shape[-2], shape[-1]))
        y = np.load(f"target{t}.npy")
        shape = y.shape
        y = y.reshape((shape[0], y.shape[2], -1))
        self.y, self.v, self.atr, self.high, self.low = \
            y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4]

        self.train_step = np.arange(0, int(self.x.shape[1] - self.x.shape[1] * 0.2 - self.step_size), self.step_size)
        # self.train_step = np.arange(0, int(self.x.shape[1] - self.x.shape[1] * 0.2 - self.step_size))
        self.test_step = self.train_step[-1] + self.step_size, self.x.shape[1] - self.step_size
        self.test_step2 = np.arange(self.test_step[0], self.test_step[1], self.step_size)


    def train(self, b = 128):
        tree_idx, replay, isw = self.memory.sample(b)

        self.states = states = np.array([a[0][0] for a in replay], np.float32)
        new_states = np.array([a[0][3] for a in replay], np.float32)
        actions = np.array([a[0][1] for a in replay]).reshape((-1, 1))
        rewards = np.array([a[0][2] for a in replay], np.float32).reshape((-1, 1))
        gamma = np.array([a[0][4] for a in replay]).reshape((-1, 1))

        self.tau = tau = np.random.uniform(0, 1, (len(tree_idx), 32))
        target_tau = np.random.uniform(0, 1, (len(tree_idx), 32))

        target_q = self.target_q([new_states, target_tau])
        target_a = np.argmax(np.sum(self.q([new_states, tau]), -1), -1)

        with tf.GradientTape() as tape:
            q = self.model([states, tau])
            q_backup = q.numpy()

            for i in range(len(tree_idx)):
                q_backup[i, actions[i]] = rewards[i] + gamma[i] * target_q[i, target_a[i]]

            error = q_backup - q
            tau = tau.reshape((-1, 1, 32))

            huber = tf.where(abs(error) <= 2, error ** 2 * .5, .5 * 2 ** 2 + 2 * tf.abs(error) - 2)
            loss = tf.maximum(tau * huber, (tau - 1) * huber)

            error = tf.reduce_sum(tf.reduce_sum(loss, 1), -1)
            loss = tf.reduce_mean(error)
            # loss = tf.reduce_mean(error * isw)

        self.e.append(loss)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # gradients = [tf.clip_by_value(g, -1, 1) for g in gradients]
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        ae = error.numpy().reshape((-1,))
        self.ae = ae
        self.memory.batch_update(tree_idx, ae)

        self.target_model.set_weights(0.005 * np.array(self.model.get_weights()) + 0.995 * np.array(self.target_model.get_weights()))


    def step(self, types=0):
        train = True if types == 0 else False
        step = range(25) if train else range(10)
        self.exp = []

        for _ in step:
            s = 0
            if types == 2:
                h = np.random.randint(self.test_step[0], self.test_step[1])
            else:
                h = np.random.choice(self.train_step)

            self.df = df = self.x[s, h:h + self.step_size]
            self.trend = trend = self.y[s, h:h + self.step_size]
            v = self.v[s, h:h + self.step_size]

            if not train:
                old_a = 0
                lot = 0
                money = self.money
                self.pip = []

                tau = np.random.uniform(0, 1, (self.step_size, 32))
                q = self.q([df, tau])
                q = np.mean(q, -1) / (np.sqrt(np.std(q, -1)) + 1e-10)
                self.a = action = np.argmax(q, -1)
                # action = np.argmax( np.sum( self.q([df, tau]), -1 ), -1)

                for idx, action in zip(range(len(trend) - 1), action):
                    action = 0 if action == 0 else -1 if action == 1 else 1

                    if (action == 1 or action == -1) and lot == 0:
                        lot = (money * 0.05 / (trend[idx] / self.leverage))

                    r = trend[idx + 1] - trend[idx]
                    r = (action * r - self.spread * np.abs(old_a - action)) * lot
                    money += r
                    money = np.clip(money, 0, None)
                    self.pip.append(r)
                    if old_a != action:
                        lot = 0

                    if money <= 0:
                        break

                    old_a = action

                g = ((money - self.money) / self.money) * 100

                self.exp.append(g)

            else:
                gammas = []
                position = 0
                actions = []
                rewards = []
                old_a = 0
                noise_w = [w + np.random.normal(0, self.sigma, w.shape) for w in self.w]
                noise = np.random.normal(0, 0.1, self.action_size)
                self.model.set_weights(noise_w)

                for idx in range(len(trend) -1):
                    df_t = np.array([df[idx]])
                    df_t = np.random.normal(df_t, 0.005)
                    if np.random.rand() > 0.1:
                        tau = np.random.uniform(0, 1, (1, 32))
                        q = self.q([df_t, tau])
                        q = np.mean(q, -1)
                        action = np.argmax(q, -1)[0]
                    else:
                        tau = np.random.uniform(0, 1, (1, 32))
                        q = self.q([df_t, tau])
                        q = np.mean(q, -1)
                        q = np.abs(q) / np.sum(np.abs(q), 1).reshape((-1, 1)) * (np.abs(q) / q)
                        q += noise
                        action = np.argmax(q, -1)[0]


                    action = int(action)
                    actions.append(action)
                    action = action if action == 0 else -1 if action == 1 else 1

                    if old_a == action:
                        r = 0
                        # r = trend[idx + 1] - trend[idx]
                        # r = action * r - self.spread * np.abs(old_a - action)
                        gamma = 0.99
                    elif position != 0:
                        r = trend[idx + 1] - position
                        r = action * r - self.spread# * np.abs(old_a - action)
                        gamma = 0
                        position = 0
                    else:
                        r = 0
                        gamma = 0.99

                    if (action == -1 or action == 1) and position == 0:
                        position = trend[idx]

                    gammas.append(gamma)
                    rewards.append(r)

                    old_a = action

                    if len(rewards) > self.n:
                        r = np.sum(rewards[-self.n:]) * 0.99 ** self.n
                        if gammas[idx - (self.n - 1)] == 0.99 and 0 in gammas[-self.n:]:
                            gammas[idx - (self.n - 1)] = 0.1
                        try:
                            e = df[idx - (self.n - 1)], actions[idx - (self.n - 1)], r, df[idx + self.n], gammas[
                                idx - (self.n - 1)]
                            self.memory.store(e)
                            if (self.restore + 1) % 64 == 0:
                                self.model.set_weights(self.w)
                                self.train()
                                self.w = self.model.get_weights()
                                noise = np.random.normal(0, 0.1, self.action_size)
                            self.restore += 1
                        except:
                            pass

                    if (idx + 1) % (self.step_size // 2) == 0:
                        # 計算コストが高い
                        self.epsilon = np.clip(self.epsilon * 0.99999, 0.05, None)
                        self.threshold = -np.log(1 - self.epsilon + self.epsilon / self.action_size)
                        self.model.set_weights(self.w)
                        q = self.q([self.states, self.tau])
                        q = tf.reduce_mean(q, -1)
                        noise_w = [w + np.random.normal(0, self.sigma, w.shape) for w in self.w]
                        self.model.set_weights(noise_w)
                        qe = self.q([self.states, self.tau])
                        qe = tf.reduce_mean(qe, -1)

                        kl = tf.reduce_sum(
                                        tf.nn.softmax(q) * (
                                        tf.math.log(tf.nn.softmax(q) + 1e-10) - tf.math.log(tf.nn.softmax(qe) + 1e-10)),
                                        axis=-1)

                        mean_kl = np.mean(kl.numpy())
                        self.sigma = self.alpha * self.sigma if mean_kl < self.threshold else 1 / self.alpha * self.sigma
                        noise_w = [w + np.random.normal(0, self.sigma, w.shape) for w in self.w]
                        self.model.set_weights(noise_w)

                self.i += 1
        if train:
            self.model.set_weights(self.w)

    def run(self):
        train_h = []
        test_h = []
        for idx in range(10000):
            start = time.time()
            if idx % 10 == 0:
                self.h = np.random.choice(self.train_step)
            self.step(0)

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
