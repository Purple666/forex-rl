# forex-rl
fotrx trading reinforcement learning

example

```
cd forex_rl
python rl/agent/dqn.py
```

```python
cd forex_rl

python

import rl.agent.dqn as dqn
agent = dqn()

#train
agent.run()

#test
agent.step(1) #1 = train_sample, 2 = test_sample
