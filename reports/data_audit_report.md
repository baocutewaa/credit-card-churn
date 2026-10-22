# Báo cáo Kiểm định Dữ liệu & Chất lượng (Phase 1 Data Quality Audit)

> **Dự án:** Credit Card Churn Prediction  
> **Tập dữ liệu:** `data/raw/train.csv` (12.120 dòng thô)  
> **Thời điểm snapshot:** `2026-06-30`  
> **Quy chuẩn thực hiện:** Tuân thủ 100% mục tiêu tại [Classification_Plan.md](file:///c:/Users/Admin/Desktop/Kỳ%20Thu%202026/T4.3.%20Ứng%20Dụng%20AI%20Trong%20KD%20Và%20Quản%20Lý/Kaggle_Competition/credit-card-churn/Classification_Plan.md)

---

## 1. Kiểm tra Cấu trúc và Schema (Schema Audit)

- **Số dòng ban đầu:** 12,120 dòng.
- **Số cột:** 24 cột.
- **Tất cả cột bắt buộc:** ĐẦY ĐỦ.
- **Cột hằng số phát hiện:**
  - `bank_code`: 100% là `VB01` (đơn trị, cần loại khỏi mô hình).
  - `currency`: 11.940 `VND` và 60 `vnd` (sau khi viết hoa 100% là `VND`, cần loại khỏi mô hình).

---

## 2. Kiểm tra Trùng lặp & Khử trùng lặp (Deduplication)

Dữ liệu thô có **120 dòng trùng lặp hoàn toàn** trên tất cả các cột.

| Chỉ số | Giá trị | Ý nghĩa nghiệp vụ |
|---|---:|---|
| **Tổng số dòng thô ban đầu** | 12,120 | Kích thước file gốc từ ban tổ chức |
| **Số dòng trùng lặp hoàn toàn** | **120** | Bản sao y hệt trên toàn bộ 24 cột |
| **Số khách hàng duy nhất sau dedup** | **12,000** | Giữ trọn 100% khách hàng sạch |
| **Trùng lặp ID còn lại sau dedup** | **0** | Mỗi `customer_id` ứng với duy nhất 1 khách hàng |

> [!IMPORTANT]
> Việc loại bỏ chính xác 120 dòng trùng lặp trước khi chia fold giúp bảo vệ mô hình khỏi hiện tượng tăng trọng số ảo và ngăn chặn rò rỉ dữ liệu qua các validation folds.

---

## 3. Phân phối Target `churn_3m`

Sau khi khử trùng lặp (12.000 khách hàng sạch):

- **Số khách hàng không churn (`churn_3m = 0`):** 10,595 (88.292%).
- **Số khách hàng rời bỏ (`churn_3m = 1`):** 1,405 (11.708%).
- **Tính hợp lệ:** Toàn bộ target nằm trong tập `{0, 1}`, không có giá trị null hay nhãn dị biệt.
- **Đánh giá mất cân bằng lớp:** Mức độ imbalanced ~1:7.5. Do đó, metric tối ưu là **ROC-AUC** kết hợp PR-AUC, không dùng Accuracy đơn thuần.

---

## 4. Kiểm tra Giá trị Khuyết thiếu (Missingness Audit)

Tình trạng missing values trên 12.000 khách hàng sạch:

| Tên Cột | Số lượng thiếu | Tỷ lệ thiếu (%) |
|---|---:|---:|
| `other_bank_cards` | 2,446 | 20.38% |
| `income_month` | 1,547 | 12.89% |
| `app_logins_30d` | 1,504 | 12.53% |
| `salary_via_bank` | 712 | 5.93% |
| `occupation` | 608 | 5.07% |
| `last_contact_channel` | 508 | 4.23% |
| `age` | 388 | 3.23% |
| `gender` | 229 | 1.91% |
| `province` | 197 | 1.64% |
| `open_date` | 78 | 0.65% |

**Quy tắc xử lý:**
1. Không loại bỏ bất kỳ dòng nào chỉ vì thiếu dữ liệu.
2. Với biến số: Dùng `MedianImputer` kết hợp tạo `MissingIndicator` (`_isna`) cho các cột có tỷ lệ thiếu cao (`other_bank_cards`, `income_month`, `app_logins_30d`, `age`).
3. Với biến phân loại: Gán nhãn `Unknown` cho các giá trị null (`gender`, `province`, `occupation`, `salary_via_bank`).
4. Toàn bộ imputation được fit độc quyền **bên trong từng fold** trong `ColumnTransformer`.

---

## 5. Chuẩn hóa Phân loại (Categorical Normalization)

Kiểm tra định dạng và biến thể viết hoa/thường, khoảng trắng:

- **`gender` (229 null):**
  - Xuất hiện biến thể: `F`, `F `, `female`, `M`, `male`, `m`, `Nam`, `Nữ`.
  - Chuẩn hóa: Ánh xạ chuẩn về 2 lớp chính: `Female` và `Male`, gán `Unknown` cho giá trị thiếu.
- **`province` (197 null):**
  - Xuất hiện nhiều biến thể: `TP.HCM`, `TP HCM`, `Ho Chi Minh`, `tp.hcm`, `HCM`, `Hà Nội`, `HN`, `ha noi`.
  - Chuẩn hóa: Regex map chính xác về các thực thể chuẩn hóa (`TP.HCM`, `Hà Nội`, `Đà Nẵng`, `Hải Phòng`,...).
- **`occupation` (608 null):**
  - Biến thể: `salaried`, `SALARIED`, `Salaried`, `self employed`, `Self-employed`, `Civil servant`, `civil_servant`.
  - Chuẩn hóa: Trim khoảng trắng, lowercase, ánh xạ thành `salaried`, `self_employed`, `civil_servant`, `retired`, `student`, `other`.
- **`last_contact_channel` (508 null):**
  - Có 4 kênh: `app`, `branch`, `hotline`, `none`.
  - *Lưu ý quan trọng:* Kênh `none` (3.404 khách hàng) thể hiện khách hàng không có tương tác, **không được coi là missing**!
- **`salary_via_bank` (712 null):**
  - Chuẩn hóa: `Y`, `N` và `Unknown`.

---

## 6. Kiểm tra Giá trị Dị biệt & Sai lệch Đơn vị (Outliers & Unit Audit)

| Đặc trưng | Hiện tượng phát hiện | Số lượng | Giải pháp xử lý |
|---|---|---:|---|
| **`age`** | Min=0, Max=999; có 9 bản ghi bằng 999 (sentinel), 40 bản ghi > 100, 29 bản ghi < 18 | 78 bản ghi dị biệt | Thay thế `age > 120` và `age < 18` bằng `NaN`, sau đó xử lý bằng median imputer + missing indicator |
| **`income_month`** | Median=17.7 triệu VNĐ, nhưng Max=50.500.000; có 106 bản ghi ghi nhận theo đơn vị VNĐ thay vì triệu VNĐ | 106 bản ghi | Chia cho $10^6$ đối với các giá trị $> 1.000$ để đưa về đúng đơn vị triệu VNĐ |
| **`credit_limit`** | Median=40.0 triệu VNĐ, nhưng Max=385.000.000; có 245 bản ghi ghi nhận theo đơn vị VNĐ thay vì triệu VNĐ | 245 bản ghi | Chia cho $10^6$ đối với các giá trị $> 1.000$ để đưa về đúng đơn vị triệu VNĐ |
| **`utilization_6m`** | Có 40 giá trị âm (-1.0, -0.5) và 45 giá trị > 1.0 (lên đến 15.0) | 85 bản ghi | Clip giá trị về khoảng hợp lý `[0.0, 2.0]` hoặc dùng robust feature engineering |
| **`txn_amount_3m`** | Có 169 giá trị âm (từ -0.12 đến -2.99) | 169 bản ghi | Giữ nguyên phản ánh nghiệp vụ hoàn tiền/điều chỉnh giao dịch (refunds/adjustments) |

---

## 7. Phân tích Định dạng Ngày & Ranh giới Snapshot (Date Audit)

- **Định dạng `open_date`:** Phân tích thành công 100% non-null dates qua 3 định dạng:
  - `%Y-%m-%d` (ví dụ: `2026-04-09`)
  - `%Y%m%d` (ví dụ: `20200820`)
  - `%d/%m/%Y` (ví dụ: `06/01/2024`)
- **Số lượng missing:** 78 giá trị missing được giữ lại và xử lý qua missing indicator.
- **Ranh giới Snapshot (`2026-06-30`):**
  - Toàn bộ 11.922 ngày mở thẻ hợp lệ đều diễn ra trước hoặc trong ngày `2026-06-30`.
  - **Không có ngày nào sau snapshot** (0 data leakage vi phạm thời gian).
  - Khoảng thời gian mở thẻ: từ `2011-06-21` đến `2026-06-12`.

---

## 8. Kết luận nghiệm thu Phase 1

1. Toàn bộ 120 dòng trùng lặp đã được phát hiện và loại bỏ chính xác; dữ liệu còn đúng **12.000 khách hàng duy nhất**.
2. Đã xác định rõ bản chất sai lệch đơn vị của `income_month` và `credit_limit` (ghi nhận bằng VNĐ thô thay vì triệu VNĐ).
3. Đã làm sạch toàn bộ alias văn bản và xây dựng bảng ánh xạ cố định.
4. Dữ liệu sạch đã được export tại [data/processed/train_cleaned.csv](file:///c:/Users/Admin/Desktop/Kỳ%20Thu%202026/T4.3.%20Ứng%20Dụng%20AI%20Trong%20KD%20Và%20Quản%20Lý/Kaggle_Competition/credit-card-churn/data/processed/train_cleaned.csv).
5. Sẵn sàng 100% để bước sang Phase 2 (EDA chuyên sâu) và Phase 3 (Feature Engineering).
