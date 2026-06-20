import os

# Lấy đường dẫn của thư mục hiện tại chứa file này
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Các thư mục con sẽ nằm ngay trong thư mục dự án
DATA_DIR = '/content/drive/MyDrive' 
MODEL_DIR = '/content/drive/MyDrive/models'

# Tự động tạo thư mục nếu chưa có
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
