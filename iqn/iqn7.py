from iqn6 import Agent as iqn_agent
import numpy as np


class Agent(iqn_agent):
    gamma = 0.7

    def step(self, types=0):
        n = self.n - 1
        self.b = 64
        end = self.step_size - 2
        train = True if types == 0 else False
        step = range(5) if train else range(50)
        # step = range(10) if train else range(50)
        self.exp = []

        step_size = self.step_size #if not train else self.step_size // 2
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
                old_idx = 0
                lot = 0
                position = 0
                money = self.money
                old_money = money
                self.mem = memory = []
                self.rew = rew = []
                self.a_ = actions = [None for _ in range(step_size)]

                for idx in range(step_size - 5):
                    if actions[idx] is not None:
                        a = actions[idx]
                    else:
                        if self.reset <= (2 * self.b):
                            if idx == 0:
                                tau = np.random.uniform(0, 1, (step_size, 32))
                                q = np.mean(self.q([df, tau]), -1)
                                actions = np.argmax(q, -1)
                                random_a = np.random.randint(self.action_size, size=step_size)
                                epsilon = 0.05 if self.restore is True else 0.3
                                actions = [a1 if np.random.rand() >= epsilon else a2 for a1, a2 in zip(actions, random_a)]
                            a = actions[idx]
                        else:
                            if np.random.rand() >= 0.05:
                                tau = np.random.uniform(0, 1, (1, 32))
                                df_ = np.array([df[idx]])
                                q = np.mean(self.q([df_, tau]), -1)
                                a = np.argmax(q, -1)[0]
                            else:
                                a = np.random.randint(self.action_size)
                            actions[idx] = a

                    a = 0 if a == 0 else -1 if a == 1 else 1
                    if old_a != a:
                        if a != 0:
                            for i_ in range(1, np.random.randint(3, 6)):
                                actions[idx + i_] = actions[idx]

                        if idx != 0:
                            r = old_a * (trend[idx] - position) - self.spread * np.abs(old_a)
                            r = np.clip(r, -atr[old_idx], atr[old_idx] * 2) * lot
                            money += r
                            r = ((money - old_money) / old_money)# * 100
                            if money < 0:
                                break
                            e = [df[idx], actions[idx], r, df[idx + 1], self.n-1]
                            rew.append(r)
                            memory.append(e)

                            if len(rew) >= self.n:
                                e = memory[-self.n]
                                r = np.sum(rew[-self.n:-1] * self.n_)
                                e[0][2] = r
                                e[0][3] = memory[-1][0][0]
                                self.memory.append(e)
                                self.reset += 1
                                if self.reset > (2 * self.b):
                                    self.i += 1
                                    self.train(None, self.b)

                        old_idx = idx
                        position = trend[idx]
                        lot = money * self.ar / (position / self.leverage)
                        old_money = money
                    old_a = a

                if len(self.memory) > self.max_size:
                    self.memory = self.memory[-self.max_size:]

