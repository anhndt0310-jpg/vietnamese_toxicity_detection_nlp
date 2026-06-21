import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch

# DATA_DIR = './data'
DATA_DIR = '/content/drive/MyDrive/data' # Cập nhật đường dẫn đến thư mục chứa dữ liệu trên Google Drive

# PhoBERT Tokenizer
PHOBERT_TOKENIZER = AutoTokenizer.from_pretrained("vinai/phobert-base")

def load_processed_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'df_train_clean.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'df_test_clean.csv'))

    # Đổi tên cột 'Toxicity' thành 'label' để phù hợp với quy ước của Hugging Face Transformers
    train = train.rename(columns={'Toxicity': 'label'})
    test = test.rename(columns={'Toxicity': 'label'})

    X_train = train['content_clean'].tolist()
    y_train = train['label'].tolist()
    X_test = test['content_clean'].tolist()
    y_test = test['label'].tolist()

    return X_train, X_test, y_train, y_test

def tokenize_function(texts, labels):
    tokenized_inputs = PHOBERT_TOKENIZER(texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    return {
        'input_ids': tokenized_inputs['input_ids'].squeeze(),
        'attention_mask': tokenized_inputs['attention_mask'].squeeze(),
        'labels': torch.tensor(labels)
    }
