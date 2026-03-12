import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class StockDataset(Dataset):
    # convert to tensor
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    # show length of data
    def __len__(self):
        return len(self.X)
    # return x(data) and y(label) at index i
    def __getitem__(self, i):
        return self.X[i], self.y[i]

def load_data(cache, symbol, year):
    df = cache.load(symbol, year)
    if df is None or df.empty:
        raise ValueError(f"No data found for {symbol} in {year}")
        
    features = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
    df = df[features].dropna()
    # df = df[features].ffill().bfill() 
    return df
        

def preprocess_data(df, seq_length=20):
    
    data = df.values.astype(np.float32) # convert to numpy

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = create_sequences(scaled_data, seq_length)

    return X, y, scaler # scaler : save the normalized ratio, (ex) predicted_price = scaler.inverse_transform(y_pred)


def create_sequences(data, seq_length):
    xs = []
    ys = []
    #sliding window
    for i in range(len(data) - seq_length):
        x = data[i : (i + seq_length)]
        y = data[i + seq_length, 3] # features[3] = "close"
        xs.append(x)
        ys.append(y)
    
    return np.array(xs), np.array(ys).reshape(-1, 1)

def get_stock_loader(X, y, batch_size, shuffle=False):
    dataset = StockDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)