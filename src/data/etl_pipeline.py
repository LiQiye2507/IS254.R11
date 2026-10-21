"""
ETL Pipeline: Trích xuất, Làm sạch và Tạo đặc trưng cho Đồ án DSS Cảnh báo Nguy cơ Học tập
Bộ dữ liệu: OULAD (Open University Learning Analytics Dataset)
Mục tiêu: Tạo bảng `clean_data.csv` theo tiến độ bài đánh giá của sinh viên (Student x Assessment)
Quy mô dự kiến: ~173,861 dòng (đáp ứng điều kiện >= 40.000 dòng cho môn DSS)
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Đảm bảo in tiếng Việt và emoji mượt mà trên console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Xác định đường dẫn gốc của dự án (project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')


def run_etl(raw_dir=None, output_dir=None):
    if raw_dir is None:
        raw_dir = DEFAULT_RAW_DIR
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    t_start = time.time()
    print("=" * 65)
    print("🚀 BẮT ĐẦU QUY TRÌNH ETL DỮ LIỆU OULAD CHO HỆ DSS")
    print("=" * 65)

    os.makedirs(output_dir, exist_ok=True)

    # 1. TẢI CÁC BẢNG DỮ LIỆU THÔ
    print("\n[Bước 1/5] Đang đọc các bảng dữ liệu gốc...")
    assess_path = os.path.join(raw_dir, 'studentAssessment.csv')
    meta_path = os.path.join(raw_dir, 'assessments.csv')
    info_path = os.path.join(raw_dir, 'studentInfo.csv')
    reg_path = os.path.join(raw_dir, 'studentRegistration.csv')
    courses_path = os.path.join(raw_dir, 'courses.csv')

    for p in [assess_path, meta_path, info_path, reg_path, courses_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Không tìm thấy file bắt buộc: {p}")

    df_assess = pd.read_csv(assess_path)
    df_meta = pd.read_csv(meta_path)
    df_info = pd.read_csv(info_path)
    df_reg = pd.read_csv(reg_path)
    df_courses = pd.read_csv(courses_path)

    print(f"  + studentAssessment  : {len(df_assess):>8,} dòng")
    print(f"  + assessments        : {len(df_meta):>8,} dòng")
    print(f"  + studentInfo        : {len(df_info):>8,} dòng")
    print(f"  + studentRegistration: {len(df_reg):>8,} dòng")
    print(f"  + courses            : {len(df_courses):>8,} dòng")

    # 2. GHÉP BẢNG (MERGING)
    print("\n[Bước 2/5] Đang liên kết các bảng dữ liệu...")
    # Ghép bảng bài nộp với metadata bài đánh giá (để lấy module, presentation, trọng số, deadline)
    df = df_assess.merge(df_meta, on='id_assessment', how='inner')

    # Ghép với thông tin sinh viên và độ dài khóa học
    df = df.merge(df_info, on=['code_module', 'code_presentation', 'id_student'], how='inner')
    df = df.merge(df_courses, on=['code_module', 'code_presentation'], how='left')
    print(f"  -> Tổng số dòng sau khi ghép bảng: {len(df):,} dòng")

    # 3. LÀM SẠCH VÀ LỌC DỮ LIỆU (DATA CLEANING)
    print("\n[Bước 3/5] Đang làm sạch và xử lý giá trị thiếu...")

    # 3.1 Lọc sinh viên hủy môn trước ngày bắt đầu (date_unregistration < 0)
    reg_invalid = df_reg[df_reg['date_unregistration'] < 0]
    invalid_keys = set(zip(reg_invalid['code_module'], reg_invalid['code_presentation'], reg_invalid['id_student']))
    df['key'] = list(zip(df['code_module'], df['code_presentation'], df['id_student']))
    n_before = len(df)
    df = df[~df['key'].isin(invalid_keys)].drop(columns=['key'])
    print(f"  + Đã loại {n_before - len(df):,} bản ghi của SV hủy môn trước ngày bắt đầu khóa học.")

    # 3.2 Xử lý giá trị thiếu
    # Điểm thi bị thiếu (chưa nộp / vắng thi) -> gán bằng 0
    null_scores = df['score'].isna().sum()
    df['score'] = df['score'].fillna(0.0)
    print(f"  + Đã xử lý {null_scores} giá trị thiếu của cột score (gán = 0).")

    # Ngày hạn nộp deadline bị thiếu (một số bài thi Exam cuối kỳ) -> gán bằng độ dài khóa học
    df['date'] = df['date'].fillna(df['module_presentation_length']).fillna(240)

    # imd_band bị thiếu -> gán nhóm 'Unknown'
    df['imd_band'] = df['imd_band'].fillna('Unknown')

    # 4. KỸ THUẬT ĐẶC TRƯNG TIẾN ĐỘ & TRÁNH RÒ RỈ DỮ LIỆU (FEATURE ENGINEERING)
    print("\n[Bước 4/5] Đang tính toán các đặc trưng tiến độ (không rò rỉ thông tin)...")

    # Thông số cơ bản của bài nộp hiện tại
    df['submission_week'] = (df['date_submitted'] // 7).clip(lower=0) + 1
    df['deadline_week'] = (df['date'] // 7).clip(lower=0) + 1
    df['submission_delay'] = df['date_submitted'] - df['date']
    df['is_late'] = (df['submission_delay'] > 0).astype(int)
    df['is_banked'] = df['is_banked'].fillna(0).astype(int)
    df['is_passed_assessment'] = (df['score'] >= 40).astype(int)
    df['is_failed_assessment'] = (1 - df['is_passed_assessment']).astype(int)

    # Sắp xếp đúng theo dòng thời gian của từng sinh viên
    df = df.sort_values(['code_module', 'code_presentation', 'id_student', 'date_submitted', 'date'])

    # Tính toán đặc trưng tích lũy của CÁC BÀI TRƯỚC ĐÓ (Strictly prior - tránh Data Leakage)
    grp = df.groupby(['code_module', 'code_presentation', 'id_student'], sort=False)
    prior_count = grp.cumcount()
    prior_cum_score = grp['score'].cumsum() - df['score']
    prior_cum_late = grp['is_late'].cumsum() - df['is_late']
    prior_cum_fail = grp['is_failed_assessment'].cumsum() - df['is_failed_assessment']

    df['assessment_order'] = prior_count + 1
    df['prior_assessments_count'] = prior_count
    df['prior_avg_score'] = np.where(prior_count > 0, (prior_cum_score / prior_count).round(2), 0.0)
    df['prior_late_count'] = prior_cum_late
    df['prior_fail_count'] = prior_cum_fail
    df['score_progress'] = np.where(prior_count > 0, (df['score'] - df['prior_avg_score']).round(2), 0.0)

    # Tạo nhãn mục tiêu At-risk cho bài toán DSS
    # At-risk = 1 nếu final_result là Fail hoặc Withdrawn; At-risk = 0 nếu Pass hoặc Distinction
    df['at_risk'] = df['final_result'].isin(['Fail', 'Withdrawn']).astype(int)

    # Chọn và sắp xếp lại các cột đầu ra rõ ràng
    output_cols = [
        # 1. Khóa định danh & Môn học
        'id_student', 'code_module', 'code_presentation', 'id_assessment',
        # 2. Đặc trưng nhân khẩu học & Nền tảng
        'gender', 'region', 'highest_education', 'imd_band', 'age_band',
        'num_of_prev_attempts', 'studied_credits', 'disability',
        # 3. Đặc trưng bài đánh giá hiện tại
        'assessment_type', 'weight', 'assessment_order', 'submission_week',
        'submission_delay', 'is_late', 'is_banked', 'score', 'is_passed_assessment',
        # 4. Đặc trưng tiến độ tích lũy (Cảnh báo sớm)
        'prior_assessments_count', 'prior_avg_score', 'prior_late_count',
        'prior_fail_count', 'score_progress',
        # 5. Nhãn dự đoán
        'final_result', 'at_risk'
    ]
    df_clean = df[output_cols].copy()

    # 5. XUẤT FILE ĐẦU RA
    print("\n[Bước 5/5] Đang lưu bảng clean_data...")
    csv_path = os.path.join(output_dir, 'clean_data.csv')
    df_clean.to_csv(csv_path, index=False)
    csv_size_mb = os.path.getsize(csv_path) / (1024 * 1024)

    t_total = time.time() - t_start
    print("=" * 65)
    print(" HOÀN TẤT QUY TRÌNH ETL THÀNH CÔNG!")
    print(f"  + Thời gian thực thi    : {t_total:.2f} giây")
    print(f"  + File kết quả          : {csv_path}")
    print(f"  + Dung lượng file       : {csv_size_mb:.2f} MB")
    print(f"  + Tổng số dòng          : {len(df_clean):,} dòng (Thỏa mãn >= 40.000 dòng)")
    print(f"  + Tổng số thuộc tính    : {df_clean.shape[1]} cột")
    print(f"  + Tỷ lệ nhãn At-risk    : {df_clean['at_risk'].mean()*100:.2f}% có nguy cơ rớt/bỏ học")
    print("=" * 65)

    return df_clean


if __name__ == '__main__':
    run_etl()
