#!/bin/env/python

import pandas as pd
import numpy as np
import time

data = pd.read_csv('candle5m.csv',index_col=0,parse_dates=True)
obv_init = 0 
close_init = data['close'].values[0]
index_next = pd.Timestamp(data.index[0])
obv = 0
iterator = data.iterrows()
candles = 0
print (obv)


for index, row in iterator:
    
    try:
        index_next = next(iterator)
        index_next = index_next[0]
    except:
        print ("End of file")
        print ("Total candles:", candles)

    if index != index_next:
    
        close_value = row['close']
        volume_value = row['volume']
    
        if close_init > close_value:
            obv = obv_init - volume_value
    
        elif close_init < close_value:
            obv = obv_init + volume_value
    
        else:
            obv = 0
        
        print ("Candle Date:", index, "OBV:",obv, "Close_init:", close_init, "Close:", close_value, "Volume:", volume_value)
#        print ("Next date:", index_next)
        #print (index_next)
        candles = candles + 1
        obv_init = obv
        close_init = close_value 

    

    

