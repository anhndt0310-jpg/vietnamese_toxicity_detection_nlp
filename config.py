import os

BASE_DIR = os.path.abspath(os.getcwd()) 

# Trỏ thẳng vào thư mục data trong cùng thư mục với code
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Tự động tạo nếu chưa có
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
