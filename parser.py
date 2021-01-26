import pandas as pd
import json
import datetime
from tqdm import tqdm
import sys

try:
    arg1, arg2 = sys.argv[1], sys.argv[2]
except:
    sys.exit("Usage 'python parser.py [filename.txt] [file_output.csv]")

data = []

with open(arg1) as f:
    for line in f:
        data.append(json.loads(line))
numbers = len(data)
df=pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close','volume'])

for i in tqdm(range(numbers)):
    date = (data[i]['data']['k']['t'])/1000
    sopen = data[i]['data']['k']['o']
    high =  data[i]['data']['k']['h']
    low =  data[i]['data']['k']['l']
    close = data[i]['data']['k']['c']
    volume = data[i]['data']['k']['v']
    df = df.append({"date":date,"open":sopen,"high":high,"low":low,"close":close,"volume":volume},ignore_index=True)
    
df['date'] = df['date'].apply(lambda x:datetime.datetime.fromtimestamp(x))
print ("Writting to file...")
df.to_csv(arg2, index=False)
