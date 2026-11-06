"""
Temporal Snapshot & Sequence Builder cho Hệ DSS Cảnh báo sớm Nguy cơ Học tập (IS254.R11)
-----------------------------------------------------------------------------------------
Thiết kế theo phương án thống nhất (Cách 2 nâng cấp - Khớp 100% Đề cương DSS_plan.docx):
- Quy chiếu toàn bộ 7 môn học về trục Tuần lịch học: T in {1..20}, đánh giá tại T in {2, 4, 8, 12, 16, 20}.
- Giữ nguyên 25,814 lượt đăng ký sinh viên trong `clean_data.csv`.
- Kết hợp lịch hạn nộp từ `data/raw/assessments.csv` để bắt chính xác "dữ liệu ngầm":
    + Bài đã nộp tính đến tuần T (`submission_week <= T`) -> is_submitted = 1.
    + Bài đã qua hạn tại tuần T (`deadline_week <= T`) nhưng SV chưa nộp -> is_missed = 1 (cảnh báo bỏ bài).
    + Bài chưa tới hạn (`deadline_week > T`) -> Mask hoàn toàn (Zero Data Leakage).
- Xuất đồng bộ 3 định dạng dữ liệu từ cùng một nguồn thông tin tại tuần T:
    1. Bảng đặc trưng 2D tại từng mốc T (cho Baseline Logistic Regression / Random Forest).
    2. Bảng bằng chứng rời rạc hóa 5 tầng theo lát cắt thời gian S1..S6 (cho Dynamic Bayesian Network - pgmpy).
    3. Tensor chuỗi 3D (N, 20, F_seq) + Static features + Mask (cho Many-to-Many LSTM - PyTorch).
"""

import os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_CLEAN_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'clean_data.csv')
DEFAULT_ASSESS_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'assessments.csv')
DEFAULT_REG_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'studentRegistration.csv')

EVAL_WEEKS = [2, 4, 8, 12, 16, 20]
MAX_WEEK = 20

# 6 lát cắt thời gian khớp chính xác ranh giới với EVAL_WEEKS = [2, 4, 8, 12, 16, 20]
DBN_SLICES = [
    ('S1', 1, 2),
    ('S2', 3, 4),
    ('S3', 5, 8),
    ('S4', 9, 12),
    ('S5', 13, 16),
    ('S6', 17, 20),
]

STATIC_CAT_COLS = [
    'gender', 'region', 'highest_education', 'imd_band',
    'age_band', 'disability', 'code_module'
]
STATIC_NUM_COLS = ['num_of_prev_attempts', 'studied_credits']


def load_base_enrollments_and_schedule(
    clean_path: str = DEFAULT_CLEAN_PATH,
    assess_path: str = DEFAULT_ASSESS_PATH,
    reg_path: str = DEFAULT_REG_PATH,
):
    """
    Đọc clean_data.csv (25,814 lượt đăng ký) và lịch bài kiểm tra định kỳ (TMA, CMA) từ assessments.csv.
    """
    df_clean = pd.read_csv(clean_path)

    # 1. Trích xuất danh sách 25,814 lượt đăng ký duy nhất
    enr_cols = (
        ['code_module', 'code_presentation', 'id_student']
        + STATIC_CAT_COLS[:-1]  # code_module đã có ở đầu
        + STATIC_NUM_COLS
        + ['final_result', 'at_risk']
    )
    enrollments = (
        df_clean[enr_cols]
        .drop_duplicates(subset=['code_module', 'code_presentation', 'id_student'])
        .sort_values(['code_module', 'code_presentation', 'id_student'])
        .reset_index(drop=True)
    )
    enrollments['enr_idx'] = np.arange(len(enrollments), dtype=np.int32)

    # Ghép thêm tuần hủy môn (unreg_week) từ studentRegistration.csv nếu có (phục vụ mask cho LSTM)
    if os.path.exists(reg_path):
        df_reg = pd.read_csv(reg_path)[
            ['code_module', 'code_presentation', 'id_student', 'date_unregistration']
        ]
        enrollments = enrollments.merge(
            df_reg, on=['code_module', 'code_presentation', 'id_student'], how='left'
        )
        unreg_wk = np.where(
            enrollments['date_unregistration'].notna(),
            np.clip(enrollments['date_unregistration'].fillna(999) // 7, 0, None) + 1,
            999,
        )
        enrollments['unreg_week'] = unreg_wk.astype(np.int32)
    else:
        enrollments['unreg_week'] = 999

    # 2. Đọc lịch bài đánh giá định kỳ (loại Exam cuối kỳ để tránh nhiễu cảnh báo sớm)
    if os.path.exists(assess_path):
        df_ass = pd.read_csv(assess_path)
    else:
        # Dự phòng trích xuất từ clean_data nếu máy người dùng chưa tải raw
        df_ass = (
            df_clean[['code_module', 'code_presentation', 'id_assessment', 'assessment_type', 'weight']]
            .drop_duplicates()
        )
        sub_med = df_clean.groupby('id_assessment')['submission_week'].median() * 7 - 7
        df_ass['date'] = df_ass['id_assessment'].map(sub_med)

    df_ass = df_ass[df_ass['assessment_type'] != 'Exam'].copy()
    df_ass['date'] = df_ass['date'].fillna(240.0)
    df_ass['deadline_week'] = (np.clip(df_ass['date'] // 7, 0, None) + 1).astype(np.int32)
    df_ass['weight_eff'] = np.where(df_ass['weight'] > 0, df_ass['weight'], 1.0)
    df_ass['is_tma'] = (df_ass['assessment_type'] == 'TMA').astype(np.int32)

    # 3. Tạo bảng chéo (Enrollment x Scheduled Periodic Assessments của môn học đó)
    enr_keys = enrollments[['enr_idx', 'code_module', 'code_presentation']]
    grid = enr_keys.merge(
        df_ass[['code_module', 'code_presentation', 'id_assessment', 'assessment_type', 'is_tma', 'deadline_week', 'weight_eff']],
        on=['code_module', 'code_presentation'],
        how='inner',
    )

    # Ghép kết quả thực nộp từ clean_data.csv
    sub_cols = df_clean[
        ['code_module', 'code_presentation', 'id_student', 'id_assessment',
         'submission_week', 'submission_delay', 'is_late', 'score']
    ].merge(
        enrollments[['enr_idx', 'code_module', 'code_presentation', 'id_student']],
        on=['code_module', 'code_presentation', 'id_student'],
        how='inner',
    )

    grid = grid.merge(
        sub_cols[['enr_idx', 'id_assessment', 'submission_week', 'submission_delay', 'is_late', 'score']],
        on=['enr_idx', 'id_assessment'],
        how='left',
    )

    return enrollments, grid


def compute_snapshot_features_at_week(enrollments: pd.DataFrame, grid: pd.DataFrame, week_t: int) -> pd.DataFrame:
    """
    Tính bảng đặc trưng tích lũy (Zero Data Leakage) cho 25,814 lượt đăng ký tính đến hết tuần `week_t`.
    """
    n_enr = len(enrollments)

    # Bài đã nộp trước hoặc trong tuần T
    is_sub_by_t = grid['submission_week'].notna() & (grid['submission_week'] <= week_t)
    # Bài có hạn nộp <= tuần T
    is_due_by_t = grid['deadline_week'] <= week_t
    # Bài đã qua hạn tại tuần T nhưng sinh viên CHƯA nộp tính đến hết tuần T -> Dữ liệu ngầm bỏ bài!
    is_missed_by_t = is_due_by_t & (~is_sub_by_t)
    # Các slot đã phát sinh thông tin quan sát tại tuần T (đã nộp sớm/đúng hạn HOẶC đã quá hạn)
    is_active = is_sub_by_t | is_missed_by_t

    g_active = grid.loc[is_active].copy()
    g_active['is_sub'] = is_sub_by_t.loc[is_active].astype(np.float32)
    g_active['is_missed'] = is_missed_by_t.loc[is_active].astype(np.float32)
    g_active['eff_score'] = np.where(g_active['is_sub'] > 0.5, g_active['score'].fillna(0.0), 0.0)
    g_active['sub_score'] = np.where(g_active['is_sub'] > 0.5, g_active['score'].fillna(0.0), np.nan)
    g_active['late_sub'] = np.where((g_active['is_sub'] > 0.5) & (g_active['is_late'] == 1), 1.0, 0.0)
    g_active['fail_sub'] = np.where((g_active['is_sub'] > 0.5) & (g_active['score'] < 40.0), 1.0, 0.0)
    g_active['tma_bad'] = np.where(
        (g_active['is_tma'] == 1) & ((g_active['is_missed'] > 0.5) | (g_active['late_sub'] > 0.5)),
        1.0, 0.0
    )
    g_active['w_score'] = g_active['eff_score'] * g_active['weight_eff']
    g_active['event_week'] = np.where(
        g_active['is_sub'] > 0.5, g_active['submission_week'], g_active['deadline_week']
    )
    g_active = g_active.sort_values(['enr_idx', 'event_week', 'id_assessment'])

    # Tổng số bài đến hạn <= T cho từng enrollment
    due_series = is_due_by_t.groupby(grid['enr_idx']).sum().reindex(range(n_enr), fill_value=0).to_numpy(dtype=np.float32)

    if len(g_active) > 0:
        grp = g_active.groupby('enr_idx', sort=False)
        agg = grp.agg(
            active_count=('is_sub', 'count'),
            submitted_count=('is_sub', 'sum'),
            missed_count=('is_missed', 'sum'),
            eff_mean_score=('eff_score', 'mean'),
            sub_mean_score=('sub_score', 'mean'),
            sum_w_score=('w_score', 'sum'),
            sum_w=('weight_eff', 'sum'),
            late_count=('late_sub', 'sum'),
            fail_count=('fail_sub', 'sum'),
            tma_bad_count=('tma_bad', 'sum'),
            last_score=('eff_score', 'last'),
            first_score=('eff_score', 'first'),
        ).reindex(range(n_enr))
    else:
        agg = pd.DataFrame(index=range(n_enr))

    out = enrollments[
        ['enr_idx', 'id_student', 'code_module', 'code_presentation']
        + STATIC_CAT_COLS[:-1]
        + STATIC_NUM_COLS
        + ['at_risk']
    ].copy()

    out['due_count'] = due_series
    out['submitted_count'] = agg['submitted_count'].fillna(0.0).to_numpy(dtype=np.float32)
    out['missed_count'] = agg['missed_count'].fillna(0.0).to_numpy(dtype=np.float32)
    out['has_submitted'] = (out['submitted_count'] > 0).astype(np.float32)
    out['has_active'] = (agg['active_count'].fillna(0.0).to_numpy() > 0).astype(np.float32)
    out['submission_rate'] = np.where(
        out['due_count'] > 0,
        np.clip(out['submitted_count'] / np.maximum(out['due_count'], 1.0), 0.0, 1.5),
        1.0,
    ).astype(np.float32)
    out['sub_mean_score'] = agg['sub_mean_score'].fillna(0.0).to_numpy(dtype=np.float32)
    out['eff_mean_score'] = agg['eff_mean_score'].fillna(0.0).to_numpy(dtype=np.float32)
    sum_w = agg['sum_w'].fillna(0.0).to_numpy(dtype=np.float32)
    sum_ws = agg['sum_w_score'].fillna(0.0).to_numpy(dtype=np.float32)
    out['weighted_score'] = np.where(sum_w > 0, sum_ws / np.maximum(sum_w, 1e-6), 0.0).astype(np.float32)
    out['late_count'] = agg['late_count'].fillna(0.0).to_numpy(dtype=np.float32)
    out['fail_count'] = agg['fail_count'].fillna(0.0).to_numpy(dtype=np.float32)
    out['tma_bad_count'] = agg['tma_bad_count'].fillna(0.0).to_numpy(dtype=np.float32)
    last_s = agg['last_score'].fillna(0.0).to_numpy(dtype=np.float32)
    first_s = agg['first_score'].fillna(0.0).to_numpy(dtype=np.float32)
    out['last_score'] = last_s
    out['score_trend'] = (last_s - first_s).astype(np.float32)

    return out


def build_dbn_discretized_dataset(enrollments: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """
    Xây dựng bảng biến rời rạc hóa 5 tầng cho Mạng Bayesian Động (DBN) trên 6 lát cắt thời gian:
      S1=[1..2], S2=[3..4], S3=[5..8], S4=[9..12], S5=[13..16], S6=[17..20].
    Tại thời điểm dự đoán T, chỉ các lát cắt có end_week <= T được đưa vào bằng chứng (evidence);
    các lát cắt tương lai (> T) để trống để DBN tự động marginalize!
    """
    n_enr = len(enrollments)
    dbn_df = pd.DataFrame(index=range(n_enr))

    # Tầng 1: Nhân khẩu học (Demographic)
    dbn_df['gender'] = enrollments['gender'].astype(str)
    dbn_df['age_band'] = enrollments['age_band'].astype(str)
    dbn_df['disability'] = enrollments['disability'].astype(str)
    dbn_df['region'] = enrollments['region'].astype(str)

    # Tầng 2: Nền tảng học tập (Academic Background)
    dbn_df['highest_education'] = enrollments['highest_education'].astype(str)
    # Gộp IMD band thành 4 mức để bảng xác suất CPT cô đọng, tránh overfit
    imd_map = {
        '0-10%': 'Low_0_30', '10-20': 'Low_0_30', '10-20%': 'Low_0_30', '20-30%': 'Low_0_30',
        '30-40%': 'Mid_30_60', '40-50%': 'Mid_30_60', '50-60%': 'Mid_30_60',
        '60-70%': 'High_60_100', '70-80%': 'High_60_100', '80-90%': 'High_60_100', '90-100%': 'High_60_100',
        'Unknown': 'Unknown'
    }
    dbn_df['imd_group'] = enrollments['imd_band'].map(imd_map).fillna('Unknown')
    dbn_df['prev_attempts_bin'] = np.where(
        enrollments['num_of_prev_attempts'] == 0, '0',
        np.where(enrollments['num_of_prev_attempts'] == 1, '1', '2+')
    )
    dbn_df['credits_bin'] = np.where(
        enrollments['studied_credits'] <= 60, '<=60',
        np.where(enrollments['studied_credits'] <= 120, '61_120', '>120')
    )

    # Tầng 3: Ngữ cảnh môn học
    dbn_df['code_module'] = enrollments['code_module'].astype(str)

    # Tầng 4: Chuỗi lát cắt thời gian S1..S6 (Hành vi nộp bài, Kết quả rời rạc & Trạng thái tích lũy)
    for s_name, w_start, w_end in DBN_SLICES:
        # Hoạt động phát sinh trong cửa sổ [w_start, w_end]
        sub_in_win = grid['submission_week'].notna() & grid['submission_week'].between(w_start, w_end)
        due_in_win = grid['deadline_week'].between(w_start, w_end)
        missed_in_win = due_in_win & (~(grid['submission_week'].notna() & (grid['submission_week'] <= w_end)))
        active_in_win = sub_in_win | missed_in_win

        gw = grid.loc[active_in_win].copy()
        gw['is_sub'] = sub_in_win.loc[active_in_win].astype(int)
        gw['is_missed'] = missed_in_win.loc[active_in_win].astype(int)
        gw['eff_score'] = np.where(gw['is_sub'] == 1, gw['score'].fillna(0.0), 0.0)
        gw['is_late_sub'] = np.where((gw['is_sub'] == 1) & (gw['is_late'] == 1), 1, 0)

        if len(gw) > 0:
            w_agg = gw.groupby('enr_idx', sort=False).agg(
                n_sub=('is_sub', 'sum'),
                n_miss=('is_missed', 'sum'),
                mean_eff=('eff_score', 'mean'),
                n_late=('is_late_sub', 'sum'),
            ).reindex(range(n_enr))
        else:
            w_agg = pd.DataFrame(index=range(n_enr))

        n_sub = w_agg.get('n_sub', pd.Series(0, index=range(n_enr))).fillna(0).to_numpy()
        n_miss = w_agg.get('n_miss', pd.Series(0, index=range(n_enr))).fillna(0).to_numpy()
        mean_eff = w_agg.get('mean_eff', pd.Series(0.0, index=range(n_enr))).fillna(0.0).to_numpy()
        n_late = w_agg.get('n_late', pd.Series(0, index=range(n_enr))).fillna(0).to_numpy()

        # Rời rạc hóa điểm số theo đúng 4 ngưỡng của Đề cương (<40, 40-55, 55-70, >=70) + NoEvent / Missed
        perf_bin = np.full(n_enr, 'NoEvent', dtype=object)
        has_act = (n_sub + n_miss) > 0
        only_missed = has_act & (n_sub == 0)
        has_sub = n_sub > 0
        perf_bin[only_missed] = 'Missed_0'
        perf_bin[has_sub & (mean_eff < 40.0)] = 'F_lt40'
        perf_bin[has_sub & (mean_eff >= 40.0) & (mean_eff < 55.0)] = 'D_40_55'
        perf_bin[has_sub & (mean_eff >= 55.0) & (mean_eff < 70.0)] = 'CB_55_70'
        perf_bin[has_sub & (mean_eff >= 70.0)] = 'A_ge70'

        # Rời rạc hóa kỷ luật học tập trong cửa sổ (thay thế vai trò thái độ học tập)
        disc_bin = np.full(n_enr, 'NoEvent', dtype=object)
        disc_bin[has_act & (n_miss == 0) & (n_late == 0)] = 'OnTime'
        disc_bin[has_act & (n_miss == 0) & (n_late > 0)] = 'LateSub'
        disc_bin[has_act & (n_miss > 0)] = 'MissedDue'

        # Trạng thái tích lũy tính đến cuối cửa sổ w_end (nút trạng thái Markov t-1 -> t)
        snap = compute_snapshot_features_at_week(enrollments, grid, w_end)
        cum_state = np.full(n_enr, 'NoRecord', dtype=object)
        act_mask = snap['has_active'].to_numpy() > 0.5
        w_sc = snap['weighted_score'].to_numpy()
        m_cnt = snap['missed_count'].to_numpy()
        cum_state[act_mask & (m_cnt >= 2)] = 'Critical_MultiMiss'
        cum_state[act_mask & (m_cnt < 2) & (w_sc < 40.0)] = 'HighRisk_lt40'
        cum_state[act_mask & (m_cnt < 2) & (w_sc >= 40.0) & (w_sc < 55.0)] = 'MedRisk_40_55'
        cum_state[act_mask & (m_cnt < 2) & (w_sc >= 55.0) & (w_sc < 70.0)] = 'LowRisk_55_70'
        cum_state[act_mask & (m_cnt == 0) & (w_sc >= 70.0)] = 'Strong_ge70'
        cum_state[act_mask & (m_cnt == 1) & (w_sc >= 70.0)] = 'LowRisk_55_70'

        dbn_df[f'perf_{s_name}'] = perf_bin
        dbn_df[f'disc_{s_name}'] = disc_bin
        dbn_df[f'state_{s_name}'] = cum_state

    # Tầng 5: Biến đích
    dbn_df['at_risk'] = enrollments['at_risk'].astype(int)
    return dbn_df


def build_weekly_sequence_tensors(enrollments: pd.DataFrame, grid: pd.DataFrame, max_week: int = MAX_WEEK):
    """
    Xây dựng Tensor chuỗi 3D theo từng tuần t = 1..max_week cho mô hình Many-to-Many LSTM:
      - seq_features: float32 array kích thước (N, max_week, 12)
          Chứa cả đặc trưng phát sinh trong tuần t VÀ đặc trưng tích lũy đến tuần t
          (bao gồm cờ is_submitted_wk, newly_missed_wk, cum_missed, weighted_score...).
      - loss_mask: float32 array kích thước (N, max_week)
          Bằng 1.0 ở các tuần sinh viên còn đang theo học (week <= unreg_week),
          0.0 ở các tuần sau khi sinh viên đã chính thức rút môn (đúng mục 3.2 DSS_plan.docx).
    """
    n_enr = len(enrollments)
    n_feat = 15
    seq_x = np.zeros((n_enr, max_week, n_feat), dtype=np.float32)

    # Tính trước hoạt động phát sinh đúng tại tuần t
    for w in range(1, max_week + 1):
        sub_w = grid['submission_week'].notna() & (grid['submission_week'] == w)
        due_w = grid['deadline_week'] == w
        miss_w = due_w & (~(grid['submission_week'].notna() & (grid['submission_week'] <= w)))

        # Đặc trưng trong tuần w
        n_sub_w = sub_w.groupby(grid['enr_idx']).sum().reindex(range(n_enr), fill_value=0).to_numpy(dtype=np.float32)
        n_due_w = due_w.groupby(grid['enr_idx']).sum().reindex(range(n_enr), fill_value=0).to_numpy(dtype=np.float32)
        n_miss_w = miss_w.groupby(grid['enr_idx']).sum().reindex(range(n_enr), fill_value=0).to_numpy(dtype=np.float32)

        if sub_w.any():
            gw_sub = grid.loc[sub_w]
            sc_w = gw_sub.groupby('enr_idx')['score'].mean().reindex(range(n_enr), fill_value=0.0).to_numpy(dtype=np.float32)
            late_w = gw_sub.groupby('enr_idx')['is_late'].sum().reindex(range(n_enr), fill_value=0.0).to_numpy(dtype=np.float32)
        else:
            sc_w = np.zeros(n_enr, dtype=np.float32)
            late_w = np.zeros(n_enr, dtype=np.float32)

        # Đặc trưng tích lũy tính đến hết tuần w (không rò rỉ tương lai)
        snap_w = compute_snapshot_features_at_week(enrollments, grid, w)

        idx = w - 1
        seq_x[:, idx, 0] = (n_sub_w > 0).astype(np.float32)                  # is_submitted_this_week
        seq_x[:, idx, 1] = np.log1p(n_due_w)                                 # due_this_week (log)
        seq_x[:, idx, 2] = (n_miss_w > 0).astype(np.float32)                 # is_missed_this_week (cờ bỏ bài!)
        seq_x[:, idx, 3] = sc_w / 100.0                                      # score_this_week [0..1]
        seq_x[:, idx, 4] = np.log1p(late_w)                                  # late_this_week
        seq_x[:, idx, 5] = snap_w['has_submitted'].to_numpy()                # has_any_submission_up_to_w
        seq_x[:, idx, 6] = np.log1p(snap_w['submitted_count'].to_numpy())    # cum_submitted (log)
        seq_x[:, idx, 7] = np.log1p(snap_w['missed_count'].to_numpy())       # cum_missed (log - cực kỳ quan trọng!)
        seq_x[:, idx, 8] = snap_w['submission_rate'].to_numpy()              # cum_submission_rate
        seq_x[:, idx, 9] = snap_w['weighted_score'].to_numpy() / 100.0       # cum_weighted_score [0..1]
        seq_x[:, idx, 10] = np.log1p(snap_w['fail_count'].to_numpy())        # cum_fail_count (log)
        seq_x[:, idx, 11] = snap_w['score_trend'].to_numpy() / 100.0         # score_trend [-1..1]
        seq_x[:, idx, 12] = snap_w['eff_mean_score'].to_numpy() / 100.0      # eff_mean_score [0..1]
        seq_x[:, idx, 13] = snap_w['last_score'].to_numpy() / 100.0          # last_score [0..1]
        seq_x[:, idx, 14] = np.log1p(snap_w['tma_bad_count'].to_numpy())     # cum_tma_late_or_missed (log)

    # Mask cho hàm loss trong quá trình huấn luyện (mask các tuần sau khi SV đã hủy đăng ký)
    weeks_arr = np.arange(1, max_week + 1, dtype=np.int32)[None, :]
    unreg_arr = enrollments['unreg_week'].to_numpy(dtype=np.int32)[:, None]
    # Đảm bảo ít nhất tuần 1 luôn có mask = 1
    loss_mask = np.where((weeks_arr <= unreg_arr) | (weeks_arr == 1), 1.0, 0.0).astype(np.float32)

    return seq_x, loss_mask
