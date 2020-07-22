import random
from iqn6 import Agent as iqn_agent
import tensorflow as tf
import numpy as np


class Agent(iqn_agent):
    name = ["ebu_i", "ebu.h5"]
    gamma = 0.3
    beta = 0.6
    max_size = 1000

    def train(self, b=128):
        replay = np.array(random.sample(self.memory, 1)).reshape((-1,5))

        self.states = states = np.array([a[0] for a in replay], np.float32)
        new_states = np.array([a[3] for a in replay], np.float32)
        actions = np.array([a[1] for a in replay]).reshape((-1, 1))
        rewards = np.array([a[2] for a in replay], np.float32).reshape((-1, 1))

        self.tau = tau = np.random.uniform(0, 1, (len(actions), 32))
        target_tau = np.random.uniform(0, 1, (len(actions), 32))

        target_q = self.target_q([new_states, target_tau])
        # target_a = np.argmax(np.sum(self.q([new_states, tau]), -1), -1)

        y = np.zeros((len(actions), 32))
        y[-1] += rewards[-1]
        k = np.arange(len(actions) - 2, -1, -1)
        for k in k:
            target_q[k, actions[k+1]] = self.beta * y[k + 1] + target_q[k, actions[k + 1]] * (1 - self.beta)
            y[k] = rewards[k] + self.gamma * target_q[k, np.argmax(np.mean(target_q[k], -1))]

        tau = np.random.uniform(0, 1, (len(actions), 32))
        with tf.GradientTape() as tape:
            q = self.model([states, tau])
            q_backup = q.numpy()

            for i in range(len(actions)):
                q_backup[i, actions[i]] = y[i]

            error = q_backup - q
            tau = tau.reshape((-1, 1, 32))

            huber = tf.where(abs(error) <= 2, error ** 2 * .5, .5 * 2 ** 2 + 2 * tf.abs(error) - 2)
            loss = tf.maximum(tau * huber, (tau - 1) * huber)

            error = tf.reduce_sum(tf.reduce_sum(loss, 1), -1)
            loss = tf.reduce_mean(error)  # * 0.5

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

        step_size = self.step_size# if not train else self.step_size * 2
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
                mem = []
                end = np.zeros(len(trend))
                end[-2] = 1

                # df = np.random.normal(df, np.abs(df * 0.1))
                # trend = np.random.normal(trend, np.abs(trend * 0.05))

                if self.reset > self.max_size and self.restore is not True:

                    tau = np.random.uniform(0, 1, (step_size, 32))
                    q = np.mean(self.q([df, tau]), -1)
                    action = np.argmax(q, -1)
                    action = [a if np.random.randn() > 0.05 else np.random.randint(self.action_size) for a in action]
                else:
                    action = np.random.randint(self.action_size, size=step_size)

                # for idx in range(step_size - 1):
                #     a = action[idx]
                #     a = 0 if a == 0 else -1 if a == 1 else 1
                #
                #     if old_a != a:
                #         r = old_a * (trend[idx] - position) - self.spread
                #         rew.append(r)
                #         e = df[idx], action[idx], np.sum(rew), df[idx + 1], 1
                #         rew = []
                #         position = 0
                #     else:
                #         e = df[idx], action[idx], 0, df[idx + 1], 1
                #
                #     if (a == 1 or a == -1) and position == 0:
                #         position = trend[idx]
                #
                #     # if end[idx] == 0:
                #     #     e = df[idx], action[idx], 0, df[idx+1], 1
                #     # else:
                #     #     e = df[idx], action[idx], np.sum(rew), df[idx+1], 1
                #     mem.append(e)
                #
                #     old_a = a
                for idx in range(step_size - 1):
                    a = action[idx]
                    a = 0 if a == 0 else -1 if a == 1 else 1
                    if old_a != a:
                        if idx != 0:
                            r = old_a * (trend[idx] - position) - self.spread
                            rew.append(r)
                            # e = df[idx], action[idx], r, df[idx + 1], 1
                            e = [df[old_idx], action[old_idx], r, df[idx], 1]
                            mem.append(e)
                        old_idx = idx
                        position = trend[idx]
                    old_a = a

                # self.mem = mem
                # mem[-1][2] = np.sum(rew)
                self.memory.append(mem)
                if len(self.memory) >= self.max_size:
                    self.memory.pop(0)
                self.reset += 1

                if self.reset > (0.9 * self.max_size):
                    self.train(self.b)
                    self.i += 1
