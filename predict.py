import joblib
import os
import torch
import numpy as np
from config import MODEL_DIR
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Đường dẫn tới mô hình đã fine-tune

FINE_TUNED_PATH = "/content/drive/MyDrive/phobert_cyberbullying_final"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"⏳ Đang khởi tạo PhoBERT trên {DEVICE}...")

if os.path.exists(FINE_TUNED_PATH):
    print(f"✅ Đang nạp mô hình FINE-TUNED: {FINE_TUNED_PATH}")
    # Sử dụng AutoModelForSequenceClassification vì đây là model đã fine-tune cho task phân loại
    model = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_PATH).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_PATH)
else:
    print("❌ Không tìm thấy bản fine-tuned tại thư mục models. Vui lòng kiểm tra lại bước sao chép từ Drive.")
    exit()

def predict_comment(text):
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item() * 100

    label = "Toxic (Độc hại)" if prediction == 1 else "Non-toxic (Lành mạnh)"
    return f"Kết quả: {label} ({confidence:.2f}%)"

if __name__ == '__main__':
    try:
        comment = input("Nhập câu bình luận cần kiểm tra: ")
        if not comment:
            comment = "chào thằng ngu"
            print(f"Test với câu mặc định: {comment}")
        print(predict_comment(comment))
    except EOFError:
        # Xử lý khi chạy trong môi trường non-interactive
        test_case = "chào thằng ngu"
        print(f"Nhập câu: {test_case}")
        print(predict_comment(test_case))
