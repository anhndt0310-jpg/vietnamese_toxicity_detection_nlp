import joblib
import os
from config import MODEL_DIR
from utils import load_processed_data
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

def run_training():
    print("📦 Đang nạp dữ liệu...")
    X_train, X_test, y_train, y_test = load_processed_data()
    
    print("🔄 Đang cân bằng SMOTE...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    print("🚀 Đang huấn luyện SVM...")
    model = SVC(class_weight='balanced', probability=True)
    model.fit(X_res, y_res)
    
    save_path = os.path.join(MODEL_DIR, 'svm_model.pkl')
    joblib.dump(model, save_path)
    print(f"✅ Đã lưu mô hình tại: {save_path}")

if __name__ == '__main__':
    run_training()
