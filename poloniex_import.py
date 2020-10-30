from poloniex import Poloniex
from time import time
import pandas as pd 
polo = Poloniex()
# this should give you all the historical candle data for USDT_BTC market in 1hr candles
candles = polo.returnChartData('USDT_BTC', period=1800, start=int(time()) - 60 * 60 * 24 *30 * 3)
df = pd.DataFrame.from_dict(candles)
dates = pd.to_datetime(df['date'],unit='s')

results = pd.DataFrame(dates.apply(lambda x: x.strftime('%Y-%m-%d %I-%p')))
results['close'] = df.close
results['open'] = df.open
results['low'] = df.low
results['high'] = df.high
results['volume'] = df.volume
results.to_csv('data/test/BTC_USD_30min.csv')