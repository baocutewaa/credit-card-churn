# Credit Card Churn — Dự đoán khách hàng rời bỏ dịch vụ thẻ tín dụng

> **Kaggle Competition:** Dự đoán xác suất khách hàng đóng thẻ tín dụng (`churn_3m`) trong 3 tháng tiếp theo tính từ ngày snapshot `30/06/2026`.  
> **Chỉ số đánh giá chính:** ROC-AUC.  
> **Chiến lược Validation:** Stratified 5-Fold Cross-Validation (giữ trọn 100% dữ liệu train sạch với 12.000 khách hàng sau khi loại bỏ 120 dòng trùng lặp).

---

## 1. Cấu trúc dự án (Project Structure)

Dự án được cấu trúc theo chuẩn Data Science / ML Engineering chuyên nghiệp dựa trên [Classification_Plan.md](file:///c:/Users/Admin/Desktop/Kỳ%20Thu%202026/T4.3.%20Ứng%20Dụng%20AI%20Trong%20KD%20Và%20Quản%20Lý/Kaggle_Competition/credit-card-churn/Classification_Plan.md):

```text
credit-card-churn/
├── data/
│   ├── raw/                      # Dữ liệu gốc từ Kaggle / ban tổ chức
│   │   ├── train.csv             # Dữ liệu huấn luyện thô (12.120 dòng)
│   │   ├── sample_submission.csv # Template submission (8.000 khách hàng)
│   │   └── starter_churn.ipynb   # Notebook khởi động ban đầu
│   └── processed/                # Dữ liệu sạch, OOF predictions
├── notebooks/
│   └── 01_eda.ipynb              # Notebook EDA và kiểm định chất lượng dữ liệu
├── src/                          # Mã nguồn chính của pipeline ML
│   ├── __init__.py
│   ├── config.py                 # Cấu hình đường dẫn, hằng số, feature groups
│   ├── data_validation.py        # Kiểm tra schema, duplicate, missing, target, submission contract
│   ├── preprocessing.py          # Chuẩn hóa văn bản, xử lý sentinel, parse date, ColumnTransformer
│   ├── features.py               # Feature engineering: tenure, ratios, behavioral signals
│   ├── train.py                  # Huấn luyện Stratified 5-Fold CV (chống data leakage)
│   ├── evaluate.py               # Đo lường ROC-AUC, PR-AUC, Brier score, Lift/Recall@Top-K
│   ├── explain.py                # Phân tích Feature Importance & Error Analysis (FP/FN)
│   └── predict.py                # 5-Fold Bagging Ensemble inference & sinh file nộp bài
├── models/                       # Lưu trữ fold model artifacts (.joblib)
├── reports/                      # Báo cáo và biểu đồ
│   ├── figures/                  # Biểu đồ ROC curve, PR curve, Confusion Matrix
│   └── metrics/                  # JSON báo cáo điểm số validation chi tiết
├── submissions/                  # Lưu trữ file submission.csv nộp bài Kaggle
├── tests/                        # Unit tests kiểm thử tính đúng đắn của pipeline
│   ├── __init__.py
│   └── test_validation.py       # Test schema, parse ngày, category mapping, submission contract
├── requirements.txt              # Danh sách thư viện và phiên bản phụ thuộc
├── Classification_Plan.md        # Kế hoạch chi tiết của dự án
└── README.md                     # Tài liệu hướng dẫn sử dụng
```

---

## 2. Cài đặt môi trường

Cài đặt các thư viện phụ thuộc bằng Python (khuyến nghị Python 3.10+):

```bash
pip install -r requirements.txt
```

---

## 3. Quy trình thực thi (Pipeline Workflow)

### Bước 1: Khám phá và kiểm định dữ liệu (EDA)
Mở và chạy notebook [notebooks/01_eda.ipynb](file:///c:/Users/Admin/Desktop/Kỳ%20Thu%202026/T4.3.%20Ứng%20Dụng%20AI%20Trong%20KD%20Và%20Quản%20Lý/Kaggle_Competition/credit-card-churn/notebooks/01_eda.ipynb) để kiểm tra:
- Schema dữ liệu và loại bỏ 120 dòng trùng lặp.
- Tỷ lệ target `churn_3m` (~11.71%).
- Phân tích missing values và các giá trị cực trị (sentinel values: `age=999`, số âm `utilization_6m`).

### Bước 2: Chạy kiểm thử tự động (Unit Tests)
Chạy bộ kiểm thử để đảm bảo pipeline xử lý dữ liệu và định dạng submission hoạt động chính xác:

```bash
pytest tests/
```

### Bước 3: Huấn luyện mô hình với Stratified 5-Fold Cross-Validation
Huấn luyện mô hình và đánh giá out-of-fold (OOF) mà không làm rò rỉ dữ liệu (data leakage):

```bash
# Huấn luyện Baseline Logistic Regression
python -m src.train --model logistic_regression

# Huấn luyện LightGBM Classifier
python -m src.train --model lightgbm

# Huấn luyện CatBoost Classifier
python -m src.train --model catboost
```
*Kết quả metrics qua 5 folds và file OOF predictions sẽ được tự động lưu trong `reports/metrics/` và `data/processed/`.*

### Bước 4: Dự đoán trên tập Test & Xuất file Submission
Khi ban tổ chức tải lên file `test.csv` (8.000 khách hàng snapshot 30/06/2026) vào thư mục `data/raw/`:

```bash
python -m src.predict --test-path data/raw/test.csv --model lightgbm
```
Pipeline sẽ:
1. Tiền xử lý tập test theo đúng logic từ tập train.
2. Áp dụng **5-Fold Bagging Ensemble** (lấy trung bình xác suất dự đoán từ 5 models huấn luyện qua 5 folds).
3. Tự động kiểm tra nghiêm ngặt contract nộp bài (2 cột `customer_id`, `churn_3m`, không có NaN/Inf, xác suất [0, 1]).
4. Xuất file kết quả sẵn sàng nộp bài tại [submissions/submission.csv](file:///c:/Users/Admin/Desktop/Kỳ%20Thu%202026/T4.3.%20Ứng%20Dụng%20AI%20Trong%20KD%20Và%20Quản%20Lý/Kaggle_Competition/credit-card-churn/submissions/submission.csv).