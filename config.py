import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists('/content/drive/MyDrive'):
    BASE_DIR = '/content/drive/MyDrive/vietnamese_toxicity_detection_nlp'

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
