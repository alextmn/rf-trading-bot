import pandas as pd

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %I-%p')

raw_df= pd.read_csv('data/test/BTC_USD_1h.csv', 
    parse_dates=['date'], date_parser=dateparse)
raw_df = raw_df.sort_values(by='date').reset_index(drop=True)

print(raw_df[0:10])