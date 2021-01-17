import pandas as pd
import json
import datetime

data = []
with open('data5m.txt') as f:
    for line in f:
        data.append(json.loads(line))
numbers = len(data)
df=pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close','volume'])

for i in range(numbers):
    date = (data[i]['data']['k']['t'])/1000
    sopen = data[i]['data']['k']['o']
    high =  data[i]['data']['k']['h']
    low =  data[i]['data']['k']['l']
    close = data[i]['data']['k']['c']
    volume = data[i]['data']['k']['v']
    df = df.append({"date":date,"open":sopen,"high":high,"low":low,"close":close,"volume":volume},ignore_index=True)

df['date'] = df['date'].apply(lambda x:datetime.datetime.fromtimestamp(x))
df.to_csv('candle5m.csv', index=False)
print (df)
