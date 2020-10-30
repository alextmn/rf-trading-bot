import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class FeatureExtractor:
    def __init__(self, df):
       self.df = df
       self.open = df['open'].astype('float')
       self.close = df['close'].astype('float')
       self.high = df['high'].astype('float')
       self.low = df['low'].astype('float')
       self.volume = df['volume'].astype('float')

    def add_bar_features(self):
        #stationary candle
        self.df['bar_hc'] = self.high - self.close
        self.df['bar_ho'] = self.high - self.open
        self.df['bar_hl'] = self.high - self.low
        self.df['bar_cl'] = self.close - self.low
        self.df['bar_ol'] = self.open - self.low
        self.df['bar_co'] = self.close - self.open
        self.df['ret_mean'] = self.df['close'].rolling(50).mean() - self.df['close']
        self.df['ret_std'] = self.df['ret_mean']/self.df['ret_mean'].rolling(50).std()
        #self.df['bar_mov3'] = self.df['close'] - self.df['close'].shift(3)
        #self.df['bar_mov9'] = self.df['close'] - self.df['close'].shift(9)
        return self.df
