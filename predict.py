import torch
import joblib
import os
import argparse
from transformers import AutoModel, AutoTokenizer
from preprocess import clean_text # Hàm làm sạch bạn đã viết
from config import MODEL_DIR

# Cấu hình thiết bị
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Predictor:
    def __init__(self, model_path):
        print("Đang nạp PhoBERT để trích xuất đặc trưng...")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.model = AutoModel.from_pretrained("vinai/phobert-base").to(DEVICE)
        self.model.eval()
        
        print("Đang nạp mô hình Logistic Regression đã huấn luyện...")
        self.logreg_model = joblib.load(model_path)

    def predict(self, text):
        # 1. Làm sạch câu
        clean_input = clean_text(text)
        
        # 2. Tạo Embedding bằng PhoBERT
        inputs = self.tokenizer(clean_input, return_tensors='pt', 
                                padding='max_length', truncation=True, 
                                max_length=128).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Lấy token [CLS]
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        # 3. Dự đoán
        prediction = self.logreg_model.predict(embedding)
        return "Độc hại" if prediction[0] == 1 else "Lành mạnh"

def main():
    parser = argparse.ArgumentParser(description="Dự đoán bình luận độc hại")
    parser.add_argument('--text', type=str, required=True, help="Câu cần dự đoán")
    args = parser.parse_args()

    model_path = os.path.join(MODEL_DIR, 'phobert_logreg_smote.pkl')
    
    if not os.path.exists(model_path):
        print("Lỗi: Không tìm thấy file mô hình! Hãy chạy train trước.")
        return

    predictor = Predictor(model_path)
    result = predictor.predict(args.text)
    print(f"\nKết quả dự đoán cho: '{args.text}'")
    print(f"-> Nhãn: {result}")

if __name__ == '__main__':
    main()
