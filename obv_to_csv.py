#!/bin/env/python

import pandas as pd
import numpy as np
import time
import sys
from tqdm import tqdm
try:
    arg1, arg2 = sys.argv[1], sys.argv[2]
except:
    print("Adiciona a coluna OBV, pelos dados da Binance, ja transformados em DataFrame")
    sys.exit("Usage: 'python obv_to_csv [data.csv] [output.csv]'")

data = pd.read_csv(arg1,index_col=0,parse_dates=True)
obv_init = 0 
close_init = data['close'].values[0]
index_next = pd.Timestamp(data.index[0])
obv = 0
close_value = data['close'].values[0]
volume_value = data['volume'].values[0]
iterator = data.iterrows()
candles = 0
data_to_write = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume', 'obv'])

for index, row in tqdm(iterator,total=data.shape[0]):
    
    if index != index_next:
        
        high_value = row['high']
        low_value = row['low']
        close_value = row['close']

        if close_init > close_value:
            obv = obv_init - volume_value
    
        elif close_init < close_value:
            obv = obv_init + volume_value
    
#        print ("Candle Date:", index, "OBV:",obv, "Close_init:", close_init, "Close:", close_value, "Volume:", volume_value)
        obv_init = obv
        candles = candles + 1
        data_to_write = data_to_write.append({"date":index,"open":close_init,"high":high_value,"low":low_value,"close":close_value,"volume":volume_value,"obv":obv},ignore_index=True)
    
    volume_value = row['volume']
    close_init = close_value 
    index_next = index        

data_to_write.to_csv(arg2, index=False)
print ("Total candles: ", candles)

