import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm # Changed to standard tqdm for script execution

# Assuming utils.py is in the same directory and already updated
from utils import load_processed_data, PHOBERT_TOKENIZER # PHOBERT_TOKENIZER is now imported directly from utils

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the base PhoBERT model for embedding extraction
# This model will be used to get numerical features from text for SMOTE
PHOBERT_BASE_MODEL = AutoModel.from_pretrained("vinai/phobert-base").to(DEVICE)
PHOBERT_BASE_MODEL.eval() # Set to evaluation mode for feature extraction

# Function to get embeddings from text
def get_embeddings(texts, tokenizer, model, batch_size=32):
    all_embeddings = []
    # Use tqdm for a progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
        batch_texts = texts[i:i + batch_size]
        # Tokenize the batch
        tokenized_inputs = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=128, # Assuming max_length 128 as used in tokenize_function in utils
            return_tensors='pt'
        ).to(DEVICE)

        with torch.no_grad():
            # Get model outputs (last hidden state contains contextual embeddings)
            outputs = model(**tokenized_inputs)
            # Take the [CLS] token embedding as the sentence embedding
            # The [CLS] token is typically at index 0
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)

def run_training():
    print("Đang nạp dữ liệu...")
    X_train_raw, X_test_raw, y_train, y_test = load_processed_data()

    print(" Đang trích xuất embeddings cho dữ liệu huấn luyện...")
    X_train_embeddings = get_embeddings(X_train_raw, PHOBERT_TOKENIZER, PHOBERT_BASE_MODEL)

    print(" Đang cân bằng SMOTE trên embeddings huấn luyện...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_embeddings, y_train)

    print(f"Kích thước X_train (trước SMOTE): {X_train_raw.shape if isinstance(X_train_raw, np.ndarray) else len(X_train_raw)} văn bản")
    print(f"Kích thước X_train_embeddings: {X_train_embeddings.shape}")
    print(f"Kích thước X_train (sau SMOTE): {X_resampled.shape}")
    print(f"Phân bố nhãn (trước SMOTE):\n{pd.Series(y_train).value_counts()}")
    print(f"Phân bố nhãn (sau SMOTE):\n{pd.Series(y_resampled).value_counts()}")

    # --- Tiếp theo: Huấn luyện mô hình phân loại trên dữ liệu đã cân bằng ---
    print("\n Đã hoàn tất bước cân bằng dữ liệu bằng SMOTE trên embeddings.")



if __name__ == '__main__':
    run_training()
