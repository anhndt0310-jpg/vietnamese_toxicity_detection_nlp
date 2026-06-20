import os

# Lấy đường dẫn của thư mục hiện tại chứa file này
BASE_DIR = '/content/drive/MyDrive'

# Các thư mục con sẽ nằm ngay trong thư mục dự án
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Tự động tạo thư mục nếu chưa có
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
