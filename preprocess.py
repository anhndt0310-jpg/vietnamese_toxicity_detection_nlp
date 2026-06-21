import pandas as pd
import re
from pyvi import ViTokenizer 

def clean_text(text):
    if not isinstance(text, str): return ""
    # 1. Chuyển về chữ thường
    text = text.lower()
    # 2. Xóa link, số điện thoại, email
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # 3. Tách từ
    text = ViTokenizer.tokenize(text)
    return text

def load_and_clean_data(file_name):
    path = os.path.join(DATA_DIR, file_name)
    df = pd.read_csv(path)
    
    df['clean_comment'] = df['comment'].apply(clean_text)
    
    df = df.dropna(subset=['clean_comment'])
    
    print(f"Đã xử lý xong {len(df)} dòng dữ liệu.")
    return df
