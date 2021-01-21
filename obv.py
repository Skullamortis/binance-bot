#!/bin/env/python

import pandas as pd
import numpy as np

data = pd.read_csv('candle5m.csv',index_col=0,parse_dates=True)
obv_init = 0 
close_init = data['close'].values[0]

for index, row in data.iterrows():
    
    close_value = row['close']
    volume_value = row['volume']
    
    if close_init > close_value:
        obv = obv_init - volume_value
    
    elif close_init < close_value:
        obv = obv_init + volume_value
    
    else:
        obv = 0
    
    obv_init = obv
    close_init = close_value 
    
    print (obv)
#print (obv)

