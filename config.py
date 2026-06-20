import os

# Nếu bạn để đoạn code kiểm tra môi trường trong 1 file chung, hãy import nó
# Hoặc copy logic đó vào đây để config.py tự xác định BASE_DIR
def get_base_dir():
    # Giữ lại logic if-else của bạn ở đây
    if os.path.exists('/content/drive'):
        return '/content/drive/MyDrive/vietnamese_toxicity_detection_nlp'
    return os.path.abspath(os.getcwd())

BASE_DIR = get_base_dir()
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
