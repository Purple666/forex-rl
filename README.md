# forex-rl
fotrx trading tensorflow2(tf 2.x version) reinforcement learning

example

```
cd forex_rl
python rl/agent/dqn.py
```

```python
cd forex_rl

python

import rl.agent.dqn as dqn
import pandas as pd

agent = dqn(lr=1e-3, gamma=0.99, dueling=False, noisy=True, n=3, restore=False, restore_path="rl/save_model/")

#train
agent.run()

#test
agent.step(1) #1 = train_sample, 2 = test_sample

# roi (return on invesment)
df = pd.DataFrame({"test_list": agent.exp})
print(df.describe())

        test_list
count    5.000000
mean   -24.826886
std    135.554213
min   -260.197436
25%    -12.628688
50%     20.637042
75%     57.150688
max     70.903961
