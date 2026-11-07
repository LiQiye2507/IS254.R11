# Hệ Hỗ Trợ Ra Quyết Định (DSS) Cảnh Báo Nguy Cơ Học Tập Sinh Viên

> **Đồ án môn học:** Hệ thống Hỗ trợ Ra quyết định (*Decision Support Systems* - `IS254.R11`)  
> **Trường:** Trường Đại học Công nghệ Thông tin – ĐHQG-HCM (UIT)  
> **Giảng viên hướng dẫn:** Thầy Nguyễn Hồ Duy Trí  
> **Nhóm thực hiện (Nhóm 8):**  
> - **Bạch Hồ Bắc** – MSSV: `23520086`  
> - **Đặng Công An** – MSSV: `23520002`  

---

## 📌 Giới thiệu tổng quan

Trong môi trường giáo dục đại học và học trực tuyến, việc phát hiện chậm trễ sinh viên có nguy cơ rớt môn hoặc bỏ học (sau khi đã có điểm giữa kỳ hoặc cuối kỳ) khiến các biện pháp can thiệp của cố vấn học tập bị giảm hiệu quả.

Dự án này xây dựng hoàn chỉnh một **Hệ Hỗ trợ Ra quyết định (Decision Support System - DSS)** nhằm phát hiện sớm sinh viên có nguy cơ học tập theo tiến độ từng tuần trên bộ dữ liệu giáo dục công khai **OULAD (Open University Learning Analytics Dataset)**:
1. **5-Tier Dynamic Bayesian Network (DBN - `pgmpy`):** Học cấu trúc có ràng buộc thứ tự tầng (`HillClimbSearch` + `BIC`), suy luận xác suất trong điều kiện dữ liệu đầu kỳ còn khuyết (`VariableElimination`) và giải thích nhân quả từng yếu tố ảnh hưởng (`Delta P` — Explainable AI).
2. **Masked Many-to-Many LSTM (`PyTorch`):** Học chuỗi hành vi nộp bài, điểm số và tín hiệu bỏ nộp bài quá hạn từ tuần 1 đến tuần 20 không rò rỉ tương lai.
3. **Mô hình đối chứng (`scikit-learn`):** Random Forest & Logistic Regression trên đặc trưng tích lũy theo tuần.
4. **Hệ thống phần mềm DSS:** Đóng gói dịch vụ suy luận qua **FastAPI**, trực quan hóa qua **Streamlit Dashboard** phân loại 3 mức nguy cơ (`Low`, `Medium`, `High`) kèm khuyến nghị can thiệp sư phạm, và đóng gói bằng **Docker Compose**.

---

## 🏗️ Kiến trúc Hệ thống DSS

```mermaid
flowchart LR
    subgraph DataLayer ["1. Tầng Quản lý Dữ liệu"]
        A[OULAD Dataset] --> B[download_oulad.py]
        B --> C[etl_pipeline.py]
        C --> D[temporal_builder.py]
    end

    subgraph ModelLayer ["2. Tầng Quản lý Mô hình"]
        D --> M1[Baseline RF/LR]
        D --> M2[5-Tier DBN pgmpy]
        D --> M3[Masked LSTM PyTorch]
    end

    subgraph ServiceLayer ["3. Dịch vụ & Giao diện DSS"]
        M2 & M3 --> API[FastAPI Inference Service]
        API --> UI[Streamlit Decision Dashboard]
        UI --> Advisor[Cố vấn Học tập / Giảng viên]
    end
```

---

## 📂 Cấu trúc thư mục Repository

```text
IS254.R11/
├── .gitignore                        # Cấu hình bỏ qua dữ liệu thô >100MB, cache, venv
├── Dockerfile                        # Đóng gói container dịch vụ DSS
├── docker-compose.yml                # Khởi chạy đồng thời FastAPI (8000) & Streamlit (8501)
├── README.md                         # Tài liệu tổng quan & hướng dẫn vận hành hệ thống
├── requirements.txt                  # Danh sách thư viện Python
│
├── docs/                             # Tài liệu đề cương & báo cáo thực nghiệm
│   ├── DSS_plan.docx                 # Bản kế hoạch đồ án gốc (Word)
│   ├── DSS_plan.txt                  # Nội dung kế hoạch dạng văn bản
│   ├── experimental_results.md       # Bảng 4 chính thức (5-Fold CV) & đối chiếu H1, H2, H3
│   └── README.md                     # Tóm tắt tài liệu dự án
│
├── data/                             # Quản lý dữ liệu dự án
│   ├── README.md                     # Từ điển dữ liệu 28 thuộc tính & hướng dẫn ETL
│   ├── raw/                          # Dữ liệu OULAD gốc (tải tự động qua script)
│   └── processed/                    # Dữ liệu sạch & kết quả thực nghiệm
│       ├── clean_data.csv            # 173.861 dòng x 28 cột (~22.5 MB)
│       ├── benchmark_folds.csv       # Chi tiết kết quả từng fold trên 6 mốc tuần
│       └── benchmark_summary.csv     # Trung bình ± độ lệch chuẩn 5-Fold
│
├── models/                           # Trọng số mô hình đã huấn luyện sẵn sàng suy luận
│   ├── dbn_5tier.joblib              # Mạng 5-Tier DBN đã học cấu trúc & CPT
│   ├── lstm_masked.pt                # Trọng số Masked Many-to-Many LSTM & bộ chuẩn hóa
│   ├── baseline_rf_by_week.joblib    # Mô hình đối chứng Random Forest
│   └── sample_dbn_explanation.json   # Mẫu giải thích nhân quả DBN trên sinh viên thực tế
│
├── src/                              # Mã nguồn chính của hệ thống DSS
│   ├── data/
│   │   ├── download_oulad.py         # Tải tự động OULAD từ Figshare
│   │   ├── etl_pipeline.py           # Làm sạch & tạo đặc trưng chống rò rỉ dữ liệu
│   │   └── temporal_builder.py       # Cắt lát tuần T, bắt bài bỏ nộp quá hạn & tạo chuỗi 3D
│   ├── models/
│   │   ├── baseline.py               # Logistic Regression & Random Forest
│   │   ├── dbn_model.py              # 5-Tier DBN + Giải thích nhân quả Explainable AI
│   │   ├── lstm_model.py             # Masked Causal Many-to-Many LSTM
│   │   └── benchmark.py              # Kiểm định chéo 5-Fold StratifiedGroupKFold
│   ├── api/
│   │   ├── schemas.py                # Pydantic validation schemas
│   │   ├── service.py                # DSSInferenceEngine (Hybrid LSTM + DBN + Khuyến nghị)
│   │   └── main.py                   # REST API endpoints (/health, /cohort/overview, /predict/*)
│   └── dashboard/
│       └── app.py                    # Giao diện Streamlit Dashboard 4 phân hệ cho Cố vấn
│
└── notebooks/                        # Jupyter Notebooks trực quan hóa
    ├── 01_exploratory_data_analysis.ipynb
    ├── 05_model_evaluation.ipynb
    └── README.md
```

---

## ⚡ Hướng dẫn Cài đặt & Khởi chạy Hệ thống DSS

### 1. Cài đặt thư viện
```bash
git clone https://github.com/LiQiye2507/IS254.R11.git
cd IS254.R11
pip install -r requirements.txt
```

### 2. Tải dữ liệu gốc, chạy ETL & Huấn luyện lại (Tùy chọn — repo đã có sẵn `clean_data.csv` & `models/`)
```bash
python src/data/download_oulad.py
python src/data/etl_pipeline.py
python src/models/benchmark.py
```

### 3. Khởi chạy Dịch vụ FastAPI & Giao diện Dashboard Streamlit

**Cách A — Chạy trực tiếp trên máy cục bộ:**
```bash
# Cửa sổ Terminal 1: Khởi động Backend FastAPI (http://127.0.0.1:8000/docs)
uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Cửa sổ Terminal 2: Khởi động Giao diện Dashboard DSS (http://localhost:8501)
streamlit run src/dashboard/app.py
```
*(Lưu ý: Ngay cả khi bạn chỉ chạy một lệnh `streamlit run src/dashboard/app.py` mà không bật cửa sổ FastAPI, Dashboard sẽ tự động chuyển sang chế độ `Direct Engine` nạp trực tiếp mô hình mà không bị lỗi kết nối).*

**Cách B — Khởi chạy trọn gói bằng Docker Compose:**
```bash
docker compose up --build
```
* **Swagger API Docs:** `http://localhost:8000/docs`
* **DSS Advisor Dashboard:** `http://localhost:8501`

---

## 📊 Tóm tắt Kết quả Thực nghiệm 5-Fold (`StratifiedGroupKFold` theo `id_student`)

| Tuần | Mô hình | Accuracy | Recall (At-risk) | F1-Score | AUC-ROC |
| :-: | :--- | :-: | :-: | :-: | :-: |
| **T = 2** | **DBN (5-Tier)** | 0.5776 ± 0.0039 | **0.6754 ± 0.0262** | **0.5636 ± 0.0091** | 0.6282 ± 0.0043 |
| **T = 2** | **LSTM (Masked)** | 0.6211 ± 0.0060 | 0.5616 ± 0.0602 | 0.5439 ± 0.0243 | 0.6601 ± 0.0064 |
| **T = 8** | **DBN (5-Tier)** | 0.7282 ± 0.0103 | 0.6126 ± 0.0130 | 0.6456 ± 0.0127 | 0.7617 ± 0.0061 |
| **T = 8** | **LSTM (Masked)** | **0.7469 ± 0.0056** | 0.6400 ± 0.0402 | **0.6710 ± 0.0122** | **0.8044 ± 0.0033** |
| **T = 16** | **DBN (5-Tier)** | 0.8034 ± 0.0052 | 0.7336 ± 0.0189 | 0.7510 ± 0.0080 | 0.8605 ± 0.0075 |
| **T = 16** | **LSTM (Masked)** | **0.8136 ± 0.0045** | 0.7254 ± 0.0299 | **0.7586 ± 0.0061** | **0.8822 ± 0.0055** |
| **T = 20** | **LSTM (Masked)** | **0.8456 ± 0.0077** | **0.7857 ± 0.0169** | **0.8045 ± 0.0072** | **0.9144 ± 0.0053** |

*Chi tiết toàn bộ bảng so sánh đối chứng và phân tích kiểm chứng giả thuyết H1, H2, H3 xem tại [docs/experimental_results.md](docs/experimental_results.md).*

---

## 🗓️ Tiến độ Hoàn thành Đồ án (Roadmap)

- [x] **Giai đoạn 1:** Khảo sát dữ liệu OULAD & xây dựng kế hoạch đồ án (`docs/DSS_plan.docx`).
- [x] **Giai đoạn 2:** Hoàn thiện ETL Pipeline (`src/data/etl_pipeline.py`) & Phân tích khám phá dữ liệu (`notebooks/01_exploratory_data_analysis.ipynb`).
- [x] **Giai đoạn 3:** Huấn luyện & kiểm định chéo 5-Fold cho Baseline, **5-Tier DBN (`pgmpy`)** và **Masked LSTM (`PyTorch`)** trên 6 mốc tuần (`src/models/`).
- [x] **Giai đoạn 4:** Xây dựng dịch vụ suy luận REST API với **FastAPI** (`src/api/`).
- [x] **Giai đoạn 5:** Phát triển giao diện **Streamlit Dashboard** cảnh báo sớm, giải thích nhân quả & khuyến nghị sư phạm (`src/dashboard/app.py`).
- [x] **Giai đoạn 6:** Đóng gói **Docker & Docker Compose** (`Dockerfile`, `docker-compose.yml`).

---

## 📖 Tài liệu tham khảo

1. **OULAD Dataset:** Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning Analytics Dataset*. Scientific Data, 4, 170171. DOI: [10.1038/sdata.2017.171](https://doi.org/10.1038/sdata.2017.171)
2. **Nghiên cứu tiền đề:** Tan et al. (2026). *Early Prediction of Student At-Risk Status using Dynamic Bayesian Networks*.
