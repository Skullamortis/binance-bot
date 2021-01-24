#!/bin/env/python

import pandas as pd
import numpy as np
import time

data = pd.read_csv('candle5m.csv',index_col=0,parse_dates=True)
obv_init = 0 
close_init = data['close'].values[0]
index_next = pd.Timestamp(data.index[0])
obv = 0
close_value = data['close'].values[0]
volume_value = data['volume'].values[0]
iterator = data.iterrows()
candles = 0
print (obv)


for index, row in iterator:
    
    if index != index_next:
    
        close_value = row['close']

    
        if close_init > close_value:
            obv = obv_init - volume_value
    
        elif close_init < close_value:
            obv = obv_init + volume_value
    
        else:
            obv = 0
        
        print ("Candle Date:", index, "OBV:",obv, "Close_init:", close_init, "Close:", close_value, "Volume:", volume_value)
        candles = candles + 1
        obv_init = obv
    
    volume_value = row['volume']
    close_init = close_value 
    index_next = index        

    

