import argparse
from train import run_training
# Bạn có thể import thêm preprocess nếu cần

def main():
    parser = argparse.ArgumentParser(description="Vietnamese Cyberbullying Detection Pipeline")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'], 
                        help="Chế độ: huấn luyện (train) hoặc dự đoán (predict)")
    
    args = parser.parse_args()

    if args.mode == 'train':
        print("--- BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN ---")
        run_training()
    else:
        print("Chế độ predict đang được cập nhật...")

if __name__ == '__main__':
    main()
