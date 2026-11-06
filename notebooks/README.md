# Thư mục Notebooks

Thư mục này lưu trữ các Jupyter Notebook phục vụ quá trình phân tích khám phá dữ liệu (EDA), thử nghiệm mô hình và trực quan hóa kết quả cho đồ án DSS (`IS254.R11`).

## Danh mục Notebook
* [x] **`01_exploratory_data_analysis.ipynb`** *(Đã hoàn thành)*: Phân tích khám phá toàn diện trên `clean_data.csv` (173.861 bản ghi) — kiểm định chất lượng dữ liệu, hiện tượng thiên lệch sống sót (*Survivorship Bias*), phân tích nhân khẩu học & môn học, phổ điểm `score`, hành vi nộp trễ `is_late`, đặc trưng tích lũy cảnh báo sớm, ma trận tương quan và kiểm chứng ngưỡng rời rạc hóa cho mạng DBN.
* [x] **`05_model_evaluation.ipynb`** *(Đã hoàn thành)*: Đánh giá và trực quan hóa kết quả kiểm định chéo 5-Fold (`StratifiedGroupKFold`) của **DBN (5-Tier)**, **LSTM (Masked)** và **Baseline (RF, LR)** qua 6 mốc tuần `T ∈ {2, 4, 8, 12, 16, 20}`, kiểm chứng các giả thuyết H1, H2, H3 và biểu đồ giải thích nhân quả DBN.
