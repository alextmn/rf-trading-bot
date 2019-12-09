# rf-trading-bot
Deep reinforcement learning crypto trading bot

# Input Features
Stock Bar Attributes
```python
self.df['bar_hc'] = self.high - self.close
self.df['bar_ho'] = self.high - self.open
self.df['bar_hl'] = self.high - self.low
self.df['bar_cl'] = self.close - self.low
self.df['bar_ol'] = self.open - self.low
self.df['bar_co'] = self.close - self.open
self.df['bar_mov'] = self.df['close'] - self.df['close'].shift(1
```

# Trading Posistions
```python
LONG = 0
SHORT = 1
FLAT = 2
```

# RL Deep Model
```python
model = Sequential()
model.add(LSTM(64, input_shape=shape, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(32))
model.add(Activation('relu')) 
```

# Agent

```python
memory = SequentialMemory(limit=50000, window_length=30)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
             attr='eps', value_max=1., value_min=.2, value_test=.05, nb_steps=3000)

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=200,
                   enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy,
                   processor=NormalizerProcessor())

dqn.compile(Adam(lr=1e-3), metrics=['mae'])
```

# Results
The BTC price is split into traing and testing set. At the left the results of trading across 
multiple agents. At the right is the testing on the prices the agents have not seen. The legent shows how many trades were done by the aganet
![Training and Testing](/images/best_trading_agents.png)

# Most profitable 
![Training and Testing](/images/trading_agents.png)
# Credits
Based on the work: https://github.com/miroblog/deep_rl_trader/

another intersting project 
https://towardsdatascience.com/creating-bitcoin-trading-bots-that-dont-lose-money-2e7165fb0b29
