#!/bin/env/python

import pandas as pd
import numpy as np

data = pd.read_csv('candle5m.csv',index_col=0,parse_dates=True)
obv_init = 0 
close_init = data['close'].values[0]
index_next = np.datetime64(data.index[0])
obv = 0
iterator = data.iterrows()

for index, row in iterator:
    
    close_value = row['close']
    volume_value = row['volume']
        
    if index != index_next:
        if close_init > close_value:
            obv = obv_init - volume_value
    
        elif close_init < close_value:
            obv = obv_init + volume_value
    
        else:
            obv = 0
        
        print(obv)

    obv_init = obv
    close_init = close_value 
    index_next = np.datetime64(index)      
    

