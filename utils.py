import os
import numpy as np
import pandas as pd
from config import DATA_DIR

def load_processed_data():
    # Nạp dữ liệu CSV
    train = pd.read_csv(os.path.join(DATA_DIR, 'df_train_clean.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'df_test_clean.csv'))
    
    # Nạp embeddings PhoBERT (.npy)
    X_train = np.load(os.path.join(DATA_DIR, 'X_train_emb.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test_emb.npy'))
    
    return X_train, X_test, train['Toxicity'].values, test['Toxicity'].values
