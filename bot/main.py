import pandas as pd
from tensorflow.keras.optimizers import Adam, RMSprop
from model import Actor_Model, Critic_Model, Shared_Model
from utils import TradingGraph, Write_to_file
from datetime import datetime
from indicators import AddIndicators
from multiprocessing_env import train_multiprocessing, test_multiprocessing
from env2 import CustomEnv
from agent import CustomAgent, train_agent

if __name__ == "__main__":            
    df = pd.read_csv('./binance_minute.csv')

    lookback_window_size = 200
    test_window = 720*3 
    train_df = df[100:-test_window-lookback_window_size] 
    test_df = df[-test_window-lookback_window_size:]
    
    agent = CustomAgent(lookback_window_size=lookback_window_size, lr=0.0001, epochs=32, optimizer=Adam, batch_size = 1024, model="Dense")
    train_multiprocessing(CustomEnv, agent, train_df, num_worker = 16, training_batch_size=1024, visualize=False, EPISODES=32000)

