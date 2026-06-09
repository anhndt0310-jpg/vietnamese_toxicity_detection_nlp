import pandas as pd
import os
from config import DATA_DIR # Import đường dẫn từ file config

def load_and_clean_data(file_name):
    path = os.path.join(DATA_DIR, file_name)
    df = pd.read_csv(path)
    # ... (code xử lý của bạn) ...
    return df

if __name__ == "__main__":
    print("🚀 Đang chạy tiền xử lý...")
    # Chạy thử nghiệm khi gọi trực tiếp file này
