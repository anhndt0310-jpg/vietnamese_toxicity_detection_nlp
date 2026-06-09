# vietnamese_toxicity_detection_nlp
Giải pháp nhận diện và phân loại đa lớp các dạng ngôn từ kích động, bắt nạt và toxic trên không gian mạng dựa trên nội dung văn bản


# Vietnamese Cyberbullying Detection

Dự án phát hiện bình luận độc hại (Cyberbullying) sử dụng PhoBERT Fine-tuned.

## 📁 Cấu trúc thư mục
- `data/`: Chứa file CSV và PhoBERT embeddings (.npy)
- `models/`: Chứa trọng số mô hình đã huấn luyện
- `config.py`: Cấu hình đường dẫn hệ thống
- `predict.py`: Script dự đoán bình luận mới

## 🚀 Hướng dẫn cài đặt
1. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```
2. Tải dữ liệu và mô hình bỏ vào thư mục `data/` và `models/` tương ứng.

## 💻 Cách sử dụng
Để kiểm tra một câu bình luận:
```bash
python predict.py
