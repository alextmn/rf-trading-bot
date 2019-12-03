import pandas as pd
import talib
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
        # stationary candle
        self.df['bar_hc'] = self.high - self.close
        self.df['bar_ho'] = self.high - self.open
        self.df['bar_hl'] = self.high - self.low
        self.df['bar_cl'] = self.close - self.low
        self.df['bar_ol'] = self.open - self.low
        self.df['bar_co'] = self.close - self.open
        self.df['bar_mov'] = self.df['close'] - self.df['close'].shift(1)
        return self.df
