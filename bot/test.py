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

    lookback_window_size = 50
    test_window = 720*3 # 3 months 
    train_df = df[100:-test_window-lookback_window_size] # we leave 100 to have properly calculated indicators
    test_df = df[-test_window-lookback_window_size:]
    
    agent = CustomAgent(lookback_window_size=lookback_window_size, lr=0.00001, epochs=4, optimizer=Adam, batch_size = 128, model="Dense")
    test_multiprocessing(CustomEnv, agent, test_df, num_worker = 2, visualize=False, test_episodes=200, folder="2021_02_24_10_21", name="1257.43_Crypto_trader", comment="Dense")


