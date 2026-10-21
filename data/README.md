# THƯ MỤC DỮ LIỆU ĐỒ ÁN DSS (HỆ HỖ TRỢ RA QUYẾT ĐỊNH)

Thư mục này chứa toàn bộ dữ liệu gốc, tài liệu mô tả và tập dữ liệu sạch (`clean_data.csv`) phục vụ huấn luyện mô hình cho đề tài:  
**"Hệ Hỗ trợ Ra Quyết định (DSS) Cảnh báo Nguy cơ Học tập của Sinh viên" (Môn học: IS254.R11)**.

---

## 1. Cấu trúc thư mục dữ liệu

```text
data/
├── raw/                              # Dữ liệu thô gốc từ OULAD
│   ├── assessments.csv               # Metadata bài đánh giá (206 dòng)
│   ├── courses.csv                   # Danh sách 7 môn học và độ dài kỳ học (22 dòng)
│   ├── studentAssessment.csv         # Điểm các bài nộp của sinh viên (173.912 dòng)
│   ├── studentInfo.csv               # Thông tin nhân khẩu học & kết quả cuối (32.593 dòng)
│   ├── studentRegistration.csv       # Ngày đăng ký & ngày hủy môn (32.593 dòng)
│   ├── studentVle.csv                # Nhật ký click LMS (10,6 triệu dòng - 432 MB)
│   ├── vle.csv                       # Metadata các hoạt động trên LMS (6.364 dòng)
│   ├── oulad.zip                     # File nén bộ dữ liệu gốc (~44.5 MB)
│   └── .gitkeep
│
├── processed/                        # Dữ liệu sạch sau khi chạy ETL Pipeline
│   ├── clean_data.csv                # Bảng dữ liệu cuối cùng dùng để train model (173.861 dòng, 28 cột, ~22.5 MB)
│   └── .gitkeep
│
└── README.md                         # Tài liệu mô tả dữ liệu và từ điển đặc trưng
```

> [!NOTE]
> Scripts xử lý dữ liệu (`download_oulad.py` và `etl_pipeline.py`) được đặt tại thư mục `src/data/` theo chuẩn cấu trúc mã nguồn dự án.

---

## 2. Quy mô tập dữ liệu huấn luyện (`clean_data.csv`)

* **Tổng số dòng:** **173.861 dòng** (Thỏa mãn và vượt xa yêu cầu tối thiểu **40.000 dòng** của môn học).
* **Tổng số cột thuộc tính:** **28 cột**.
* **Giá trị thiếu (Missing values):** **0** (Đã được xử lý 100%).
* **Tỷ lệ nhãn mục tiêu:**
  * Nhãn `at_risk = 0` (Đậu / Xuất sắc): **76.12%** (132.348 bài nộp)
  * Nhãn `at_risk = 1` (Có nguy cơ Rớt / Bỏ học): **23.88%** (41.513 bài nộp)

---

## 3. Quy trình xử lý dữ liệu trong `src/data/etl_pipeline.py`

Quy trình ETL được thiết kế tự động qua 5 bước chính:

1. **Trích xuất (Extract):** Tải các bảng `studentAssessment`, `assessments`, `studentInfo`, `studentRegistration`, `courses`.
2. **Liên kết bảng (Relational Join):** Ghép các bài nộp với metadata bài thi (`id_assessment`), sau đó ánh xạ vào sinh viên theo bộ khóa `(code_module, code_presentation, id_student)`.
3. **Làm sạch (Clean):**
   * Lọc bỏ hoàn toàn các sinh viên đã hủy môn trước ngày bắt đầu học kỳ (`date_unregistration < 0`).
   * Xử lý điểm `score` bị thiếu bằng $0.0$.
   * Xử lý hạn nộp `date` bị thiếu bằng độ dài khóa học `module_presentation_length`.
   * Gán giá trị thiếu của `imd_band` thành `'Unknown'`.
4. **Kỹ thuật tạo đặc trưng (Feature Engineering - Chống Data Leakage):**
   * Tính tuần nộp bài, độ trễ nộp bài (`submission_delay`), cờ nộp trễ (`is_late`).
   * Tính toán các chỉ số tích lũy của **các bài trước đó** theo từng sinh viên (Strictly prior history):
     * Số bài đã nộp trước đó (`prior_assessments_count`).
     * Điểm trung bình các bài trước đó (`prior_avg_score`).
     * Số lần nộp trễ trước đó (`prior_late_count`).
     * Số lần thi rớt trước đó (`prior_fail_count`).
     * Độ tiến bộ điểm số (`score_progress = score - prior_avg_score`).
5. **Nạp & Lưu trữ (Load):** Xuất bảng dữ liệu hoàn chỉnh ra file `data/processed/clean_data.csv`.

---

## 4. Từ điển dữ liệu (Data Dictionary) của `clean_data.csv`

| STT | Tên cột | Kiểu dữ liệu | Ý nghĩa trong Hệ DSS |
| :---: | :--- | :---: | :--- |
| **I** | **Nhóm Định danh & Khóa** | | |
| 1 | `id_student` | Integer | Mã số định danh sinh viên |
| 2 | `code_module` | String | Mã môn học (AAA, BBB, CCC, DDD, EEE, FFF, GGG) |
| 3 | `code_presentation` | String | Mã học kỳ (ví dụ: 2013B, 2013J, 2014B, 2014J) |
| 4 | `id_assessment` | Integer | Mã bài kiểm tra / đánh giá |
| **II** | **Nhóm Nhân khẩu học & Nền tảng (Demographic)** | | |
| 5 | `gender` | Category | Giới tính sinh viên (M: Nam, F: Nữ) |
| 6 | `region` | String | Vùng địa lý sinh sống của sinh viên |
| 7 | `highest_education` | String | Trình độ học vấn cao nhất trước khi vào học |
| 8 | `imd_band` | String | Chỉ số thiếu thốn kinh tế xã hội (0-10% ... 90-100%, Unknown) |
| 9 | `age_band` | String | Nhóm tuổi của sinh viên (0-35, 35-55, 55<=) |
| 10 | `num_of_prev_attempts` | Integer | Số lần sinh viên đã từng học lại môn này |
| 11 | `studied_credits` | Integer | Tổng số tín chỉ sinh viên đang đăng ký học cùng lúc |
| 12 | `disability` | Category | Sinh viên có khuyết tật hay không (Y/N) |
| **III** | **Nhóm Bài đánh giá hiện tại (Current Assessment)** | | |
| 13 | `assessment_type` | Category | Loại bài: TMA (chấm bởi trợ giảng), CMA (trắc nghiệm máy tính), Exam (thi cuối kỳ) |
| 14 | `weight` | Float | Trọng số phần trăm của bài kiểm tra trong môn (0 – 100%) |
| 15 | `assessment_order` | Integer | Thứ tự bài nộp của sinh viên trong môn (bài thứ 1, 2, 3...) |
| 16 | `submission_week` | Integer | Tuần nộp bài của sinh viên |
| 17 | `submission_delay` | Float | Số ngày nộp lệch hạn (Âm: nộp sớm, Dương: nộp trễ hạn) |
| 18 | `is_late` | Binary (0/1) | Sinh viên nộp trễ hạn hay không (1 = Trễ, 0 = Đúng hạn) |
| 19 | `is_banked` | Binary (0/1) | Bài thi có được bảo lưu điểm từ kỳ trước hay không |
| 20 | `score` | Float | Điểm số đạt được của bài nộp hiện tại (0.0 – 100.0) |
| 21 | `is_passed_assessment` | Binary (0/1) | Bài thi có đạt điểm qua môn hay không ($\ge 40$ điểm) |
| **IV** | **Nhóm Tiến độ Tích lũy (Cumulative History - Hỗ trợ Cảnh báo Sớm)** | | |
| 22 | `prior_assessments_count` | Integer | Số lượng bài kiểm tra sinh viên đã hoàn thành trước bài này |
| 23 | `prior_avg_score` | Float | Điểm trung bình cộng của các bài đã nộp trước đó |
| 24 | `prior_late_count` | Integer | Tổng số lần nộp bài trễ trước bài này |
| 25 | `prior_fail_count` | Integer | Tổng số lần thi rớt các bài trước đó |
| 26 | `score_progress` | Float | Độ tiến bộ điểm số so với trung bình các bài trước |
| **V** | **Nhóm Nhãn Mục tiêu (Target Labels)** | | |
| 27 | `final_result` | Category | Kết quả cuối kỳ gốc (Pass, Distinction, Fail, Withdrawn) |
| 28 | `at_risk` | Binary (0/1) | **Nhãn dự đoán rủi ro:** 1 = Nguy cơ (Fail/Withdrawn), 0 = Đậu môn (Pass/Distinction) |

---

## 5. Hướng dẫn thực thi ETL Pipeline

Để tải dữ liệu và chạy quy trình tiền xử lý sinh file `clean_data.csv`, thực hiện các lệnh sau từ thư mục gốc của dự án:

```bash
# 1. Tải và giải nén bộ dữ liệu gốc (nếu chưa có trong data/raw)
python src/data/download_oulad.py

# 2. Thực thi pipeline tiền xử lý và kỹ thuật đặc trưng
python src/data/etl_pipeline.py
```

*Thời gian thực thi dự kiến: khoảng **1 – 2 giây** nhờ cơ chế xử lý vector hóa tối ưu.*

---

## 6. Hướng dẫn đọc dữ liệu trong Python để Train Model

```python
import pandas as pd
from sklearn.model_selection import GroupKFold

# 1. Đọc dữ liệu sạch
df = pd.read_csv('data/processed/clean_data.csv')
print(f"Kích thước dữ liệu: {df.shape}")

# 2. Tách biến đầu vào (X) và nhãn (y)
feature_cols = [
    'gender', 'region', 'highest_education', 'imd_band', 'age_band',
    'num_of_prev_attempts', 'studied_credits', 'disability',
    'assessment_type', 'weight', 'assessment_order', 'submission_week',
    'submission_delay', 'is_late', 'score',
    'prior_assessments_count', 'prior_avg_score', 'prior_late_count',
    'prior_fail_count', 'score_progress'
]

X = df[feature_cols]
y = df['at_risk']

# 3. Phân chia Train/Test theo nhóm sinh viên (Grouped Split - Tránh rò rỉ dữ liệu)
gkf = GroupKFold(n_splits=5)
train_idx, test_idx = next(gkf.split(X, y, groups=df['id_student']))

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

print(f"Số lượng mẫu Train: {len(X_train):,}")
print(f"Số lượng mẫu Test : {len(X_test):,}")
```
