"""
DSS Inference & Analytics Engine (`src/api/service.py`)
-------------------------------------------------------
Quản lý nạp mô hình đã huấn luyện (`DBN 5-Tier`, `Masked LSTM`, `Baseline RF`),
truy vấn dự báo cá nhân, phân tích tổng quan khóa học (Cohort Overview),
danh sách cảnh báo sớm (Watchlist) và giải thích nhân quả (Explainable AI).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.temporal_builder import (
    MAX_WEEK,
    STATIC_CAT_COLS,
    STATIC_NUM_COLS,
    build_dbn_discretized_dataset,
    build_weekly_sequence_tensors,
    compute_snapshot_features_at_week,
    load_base_enrollments_and_schedule,
)
from src.models.lstm_model import EarlyWarningLSTMNet

FACTOR_LABELS = {
    'gender': 'Giới tính',
    'age_band': 'Nhóm độ tuổi',
    'disability': 'Tình trạng khuyết tật',
    'region': 'Khu vực địa lý',
    'highest_education': 'Trình độ học vấn đầu vào',
    'imd_group': 'Điều kiện kinh tế - xã hội (IMD)',
    'prev_attempts_bin': 'Số lần từng học lại môn này',
    'credits_bin': 'Khối lượng tín chỉ đăng ký',
    'code_module': 'Đặc thù môn học (Module)',
    'perf_S1': 'Điểm đánh giá Tuần 1–2 (S1)',
    'disc_S1': 'Kỷ luật nộp bài Tuần 1–2 (S1)',
    'perf_S2': 'Điểm đánh giá Tuần 3–4 (S2)',
    'disc_S2': 'Kỷ luật nộp bài Tuần 3–4 (S2)',
    'perf_S3': 'Điểm đánh giá Tuần 5–8 (S3)',
    'disc_S3': 'Kỷ luật nộp bài Tuần 5–8 (S3)',
    'perf_S4': 'Điểm đánh giá Tuần 9–12 (S4)',
    'disc_S4': 'Kỷ luật nộp bài Tuần 9–12 (S4)',
    'perf_S5': 'Điểm đánh giá Tuần 13–16 (S5)',
    'disc_S5': 'Kỷ luật nộp bài Tuần 13–16 (S5)',
    'perf_S6': 'Điểm đánh giá Tuần 17–20 (S6)',
    'disc_S6': 'Kỷ luật nộp bài Tuần 17–20 (S6)',
}


def classify_risk_tier(prob: float, low_th: float = 0.30, high_th: float = 0.60) -> str:
    if prob < low_th:
        return "Low"
    if prob < high_th:
        return "Medium"
    return "High"


class DSSInferenceEngine:
    _instance = None

    @classmethod
    def get_instance(cls) -> "DSSInferenceEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        models_dir = os.path.join(PROJECT_ROOT, 'models')
        dbn_path = os.path.join(models_dir, 'dbn_5tier.joblib')
        lstm_path = os.path.join(models_dir, 'lstm_masked.pt')

        if not os.path.exists(dbn_path) or not os.path.exists(lstm_path):
            raise FileNotFoundError(
                "Chưa tìm thấy trọng số mô hình trong thư mục `models/`. "
                "Vui lòng chạy `python src/models/benchmark.py` trước."
            )

        # 1. Nạp dữ liệu cơ sở và tạo sẵn bảng DBN + Tensor chuỗi tuần 1..20
        self.enrollments, self.grid = load_base_enrollments_and_schedule()
        self.dbn_df = build_dbn_discretized_dataset(self.enrollments, self.grid)
        self.seq_x, _ = build_weekly_sequence_tensors(self.enrollments, self.grid, max_week=MAX_WEEK)

        # Chỉ mục tra cứu nhanh theo (code_module, code_presentation, id_student)
        self._lookup = {
            (str(r.code_module), str(r.code_presentation), int(r.id_student)): int(r.enr_idx)
            for r in self.enrollments[['code_module', 'code_presentation', 'id_student', 'enr_idx']].itertuples(index=False)
        }

        # 2. Nạp mô hình DBN 5 tầng
        self.dbn_clf = joblib.load(dbn_path)

        # 3. Nạp mô hình Masked Many-to-Many LSTM & tính sẵn ma trận xác suất (N, 20) siêu tốc
        ckpt = torch.load(lstm_path, map_location='cpu', weights_only=False)
        self.lstm_static_prep = ckpt['static_prep']
        self.lstm_seq_mean = ckpt['seq_mean']
        self.lstm_seq_std = ckpt['seq_std']
        self.lstm_pos_weight = float(ckpt['pos_weight_val'])

        X_stat = self.lstm_static_prep.transform(
            self.enrollments[STATIC_CAT_COLS + STATIC_NUM_COLS]
        ).astype(np.float32)
        X_seq_norm = ((self.seq_x - self.lstm_seq_mean) / self.lstm_seq_std).astype(np.float32)

        self.lstm_net = EarlyWarningLSTMNet(
            seq_dim=X_seq_norm.shape[-1],
            static_dim=X_stat.shape[-1],
            hidden_dim=int(ckpt.get('hidden_dim', 64)),
            num_layers=int(ckpt.get('num_layers', 2)),
            dropout=float(ckpt.get('dropout', 0.25)),
        )
        self.lstm_net.load_state_dict(ckpt['model_state_dict'])
        self.lstm_net.eval()

        log_pw = float(np.log(self.lstm_pos_weight))
        probs_chunks = []
        with torch.no_grad():
            for s in range(0, len(X_stat), 2048):
                e = s + 2048
                logits = self.lstm_net(
                    torch.from_numpy(X_seq_norm[s:e]),
                    torch.from_numpy(X_stat[s:e]),
                ).numpy()
                probs_chunks.append(1.0 / (1.0 + np.exp(-(logits - log_pw))))
        self.lstm_probs_matrix = np.vstack(probs_chunks)  # (25814, 20)

        # Bộ nhớ đệm kết quả suy luận toàn khóa của DBN và bảng snapshot theo tuần
        self._dbn_week_cache = {}
        self._snap_week_cache = {}

    def _get_snapshot_at_week(self, week_t: int) -> pd.DataFrame:
        if week_t not in self._snap_week_cache:
            self._snap_week_cache[week_t] = compute_snapshot_features_at_week(
                self.enrollments, self.grid, week_t
            )
        return self._snap_week_cache[week_t]

    def _get_dbn_probs_for_week(self, week_t: int) -> np.ndarray:
        # Quy chiếu tuần T sang lát cắt DBN gần nhất <= T (2, 4, 8, 12, 16, 20)
        eff_w = max([w for w in [2, 4, 8, 12, 16, 20] if w <= max(week_t, 2)])
        if eff_w not in self._dbn_week_cache:
            self._dbn_week_cache[eff_w] = self.dbn_clf.predict_proba_at_week(self.dbn_df, eff_w)
        return self._dbn_week_cache[eff_w]

    def get_enr_index(self, id_student: int, code_module: str, code_presentation: str) -> int:
        key = (str(code_module), str(code_presentation), int(id_student))
        if key not in self._lookup:
            raise KeyError(
                f"Không tìm thấy sinh viên id_student={id_student} tại môn {code_module} - kỳ {code_presentation}."
            )
        return self._lookup[key]

    def get_metadata_options(self) -> dict:
        courses = (
            self.enrollments[['code_module', 'code_presentation']]
            .drop_duplicates()
            .sort_values(['code_module', 'code_presentation'])
        )
        mod_pres = {}
        for mod, grp in courses.groupby('code_module'):
            mod_pres[str(mod)] = grp['code_presentation'].astype(str).tolist()
        return {
            'total_enrollments': int(len(self.enrollments)),
            'unique_students': int(self.enrollments['id_student'].nunique()),
            'modules_presentations': mod_pres,
            'max_week': MAX_WEEK,
        }

    def get_cohort_overview(
        self,
        week_t: int = 8,
        code_module: str = "ALL",
        code_presentation: str = "ALL",
        model_source: str = "hybrid",
        low_th: float = 0.30,
        high_th: float = 0.60,
        top_limit: int = 200,
    ) -> dict:
        week_t = int(np.clip(week_t, 1, MAX_WEEK))
        snap = self._get_snapshot_at_week(week_t).copy()

        p_lstm = self.lstm_probs_matrix[:, week_t - 1]
        p_dbn = self._get_dbn_probs_for_week(week_t)

        if model_source == "dbn":
            probs = p_dbn
        elif model_source == "lstm":
            probs = p_lstm
        else:
            # Hybrid DSS (RQ3): Đầu kỳ (T <= 2) ưu tiên độ ổn định/nhạy của DBN; từ T >= 3 ưu tiên chuỗi LSTM
            w_dbn = 0.60 if week_t <= 2 else 0.25
            probs = w_dbn * p_dbn + (1.0 - w_dbn) * p_lstm

        snap['prob_lstm'] = np.round(p_lstm, 4)
        snap['prob_dbn'] = np.round(p_dbn, 4)
        snap['prob_at_risk'] = np.round(probs, 4)
        snap['risk_level'] = np.where(
            probs < low_th, 'Low',
            np.where(probs < high_th, 'Medium', 'High')
        )
        snap['final_result'] = self.enrollments['final_result'].to_numpy()

        if code_module != "ALL":
            snap = snap[snap['code_module'] == code_module]
        if code_presentation != "ALL":
            snap = snap[snap['code_presentation'] == code_presentation]

        total_n = int(len(snap))
        tier_counts = snap['risk_level'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0).to_dict()

        # Phân bố theo môn học
        by_module = (
            snap.groupby(['code_module', 'risk_level'])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=['Low', 'Medium', 'High'], fill_value=0)
            .reset_index()
            .to_dict(orient='records')
        )

        # Danh sách cảnh báo sớm ưu tiên sắp xếp giảm dần theo xác suất rủi ro
        watchlist_df = (
            snap.sort_values('prob_at_risk', ascending=False)
            .head(top_limit)[
                [
                    'id_student', 'code_module', 'code_presentation',
                    'prob_at_risk', 'prob_lstm', 'prob_dbn', 'risk_level',
                    'submitted_count', 'missed_count', 'late_count',
                    'weighted_score', 'highest_education', 'num_of_prev_attempts',
                    'final_result', 'at_risk'
                ]
            ]
            .copy()
        )
        watchlist_df['weighted_score'] = watchlist_df['weighted_score'].round(2)

        return {
            'week_t': week_t,
            'code_module': code_module,
            'code_presentation': code_presentation,
            'model_source': model_source,
            'total_students': total_n,
            'tier_counts': {k: int(v) for k, v in tier_counts.items()},
            'by_module': by_module,
            'watchlist': watchlist_df.to_dict(orient='records'),
        }

    def predict_dbn(
        self,
        id_student: int,
        code_module: str,
        code_presentation: str,
        week_t: int = 8,
        low_th: float = 0.30,
        high_th: float = 0.60,
    ) -> dict:
        idx = self.get_enr_index(id_student, code_module, code_presentation)
        eff_w = max([w for w in [2, 4, 8, 12, 16, 20] if w <= max(week_t, 2)])
        exp = self.dbn_clf.explain_student(self.dbn_df.iloc[idx], week_t=eff_w, top_k=6)
        prob = float(exp['prob_at_risk'])
        factors = [
            {
                'factor': it['factor'],
                'factor_label': FACTOR_LABELS.get(it['factor'], it['factor']),
                'value': str(it['value']),
                'delta_prob': float(it['delta_prob']),
            }
            for it in exp['top_factors']
        ]
        return {
            'model': 'DBN_5Tier',
            'id_student': int(id_student),
            'code_module': str(code_module),
            'code_presentation': str(code_presentation),
            'week_t': int(week_t),
            'prob_at_risk': round(prob, 4),
            'risk_level': classify_risk_tier(prob, low_th, high_th),
            'top_factors': factors,
        }

    def predict_lstm(
        self,
        id_student: int,
        code_module: str,
        code_presentation: str,
        week_t: int = 8,
        low_th: float = 0.30,
        high_th: float = 0.60,
    ) -> dict:
        idx = self.get_enr_index(id_student, code_module, code_presentation)
        week_t = int(np.clip(week_t, 1, MAX_WEEK))
        curve = self.lstm_probs_matrix[idx, :week_t]
        prob_cur = float(curve[-1])
        trend = [
            {'week': int(w), 'prob_at_risk': round(float(curve[w - 1]), 4)}
            for w in range(1, week_t + 1)
        ]
        return {
            'model': 'LSTM_Masked',
            'id_student': int(id_student),
            'code_module': str(code_module),
            'code_presentation': str(code_presentation),
            'week_t': week_t,
            'prob_at_risk': round(prob_cur, 4),
            'risk_level': classify_risk_tier(prob_cur, low_th, high_th),
            'weekly_trend': trend,
        }

    def predict_hybrid_dss(
        self,
        id_student: int,
        code_module: str,
        code_presentation: str,
        week_t: int = 8,
        low_th: float = 0.30,
        high_th: float = 0.60,
    ) -> dict:
        idx = self.get_enr_index(id_student, code_module, code_presentation)
        week_t = int(np.clip(week_t, 1, MAX_WEEK))

        res_dbn = self.predict_dbn(id_student, code_module, code_presentation, week_t, low_th, high_th)
        res_lstm = self.predict_lstm(id_student, code_module, code_presentation, week_t, low_th, high_th)

        if week_t <= 2:
            primary = "DBN_5Tier (Ưu tiên đầu kỳ T <= 2 theo H1)"
            prob_hybrid = 0.60 * res_dbn['prob_at_risk'] + 0.40 * res_lstm['prob_at_risk']
        else:
            primary = "LSTM_Masked + Giải thích nhân quả DBN (T >= 3 theo H2 & H3)"
            prob_hybrid = 0.75 * res_lstm['prob_at_risk'] + 0.25 * res_dbn['prob_at_risk']

        risk_level = classify_risk_tier(prob_hybrid, low_th, high_th)
        enr_row = self.enrollments.iloc[idx]
        snap_row = self._get_snapshot_at_week(week_t).iloc[idx]

        # Lịch sử từng bài đánh giá của môn học đối với sinh viên này tính đến tuần T
        stu_grid = (
            self.grid[self.grid['enr_idx'] == idx]
            .sort_values(['deadline_week', 'id_assessment'])
        )
        timeline = []
        for r in stu_grid.itertuples(index=False):
            has_sub_by_t = pd.notna(r.submission_week) and (int(r.submission_week) <= week_t)
            if has_sub_by_t:
                st = "Submitted (Late)" if int(r.is_late) == 1 else "Submitted (OnTime)"
                sub_wk = int(r.submission_week)
                sc_val = round(float(r.score), 1)
            elif int(r.deadline_week) <= week_t:
                st = "MISSED (Overdue!)"
                sub_wk = None
                sc_val = 0.0
            else:
                st = "Upcoming (Not Due)"
                sub_wk = None
                sc_val = None

            timeline.append({
                'id_assessment': int(r.id_assessment),
                'assessment_type': str(r.assessment_type),
                'deadline_week': int(r.deadline_week),
                'weight': float(r.weight_eff),
                'status_at_week_t': st,
                'submission_week': sub_wk,
                'score': sc_val,
            })

        recs = self._generate_recommendations(risk_level, snap_row, enr_row, week_t)
        profile = {
            'gender': str(enr_row['gender']),
            'age_band': str(enr_row['age_band']),
            'highest_education': str(enr_row['highest_education']),
            'imd_band': str(enr_row['imd_band']),
            'disability': str(enr_row['disability']),
            'num_of_prev_attempts': int(enr_row['num_of_prev_attempts']),
            'studied_credits': int(enr_row['studied_credits']),
            'submitted_by_T': int(snap_row['submitted_count']),
            'missed_by_T': int(snap_row['missed_count']),
            'late_by_T': int(snap_row['late_count']),
            'weighted_score_by_T': round(float(snap_row['weighted_score']), 2),
            'actual_final_result': str(enr_row['final_result']),
        }

        return {
            'id_student': int(id_student),
            'code_module': str(code_module),
            'code_presentation': str(code_presentation),
            'week_t': week_t,
            'primary_predictor': primary,
            'prob_hybrid': round(float(prob_hybrid), 4),
            'prob_lstm': res_lstm['prob_at_risk'],
            'prob_dbn': res_dbn['prob_at_risk'],
            'risk_level': risk_level,
            'student_profile': profile,
            'weekly_trend': res_lstm['weekly_trend'],
            'dbn_explanations': res_dbn['top_factors'],
            'assessments_timeline': timeline,
            'advisor_recommendations': recs,
        }

    @staticmethod
    def _generate_recommendations(risk_level: str, snap_row: pd.Series, enr_row: pd.Series, week_t: int) -> list[str]:
        recs = []
        missed = int(snap_row['missed_count'])
        sub_cnt = int(snap_row['submitted_count'])
        late_cnt = int(snap_row['late_count'])
        w_score = float(snap_row['weighted_score'])

        if missed > 0:
            recs.append(
                f"🚨 CẢNH BÁO BỎ BÀI: Sinh viên đã quá hạn không nộp {missed} bài đánh giá tính đến Tuần {week_t}. "
                "Cố vấn cần liên hệ trực tiếp (email/điện thoại) để kiểm tra ý định bỏ học (Withdrawn) và hướng dẫn gia hạn nếu còn kịp."
            )
        if sub_cnt > 0 and w_score < 40.0:
            recs.append(
                f"📉 HỖ TRỢ HỌC THUẬT KHẨN CẤP: Điểm trung bình tích lũy hiện chỉ đạt {w_score:.1f}/100 (dưới ngưỡng đạt 40 của OULAD). "
                "Đề xuất xếp lịch phụ đạo 1-1 với trợ giảng (Tutor) trước kỳ kiểm tra TMA tiếp theo."
            )
        elif sub_cnt > 0 and w_score < 55.0:
            recs.append(
                f"⚠️ CỦNG CỐ KIẾN THỨC: Điểm tích lũy ở vùng sát ngưỡng ({w_score:.1f}/100). "
                "Khuyến nghị cung cấp tài liệu ôn tập trọng tâm và bài tập tự luyện CMA."
            )
        if late_cnt >= 2:
            recs.append(
                f"⏰ QUẢN LÝ THỜI GIAN: Đã nộp trễ {late_cnt} bài kiểm tra. "
                "Cần bật nhắc lịch sớm 72 giờ trước deadline cho sinh viên."
            )
        if int(enr_row['num_of_prev_attempts']) >= 1 and week_t <= 6:
            recs.append(
                f"🔄 LƯU Ý HỌC LẠI: Sinh viên từng học lại môn này ({int(enr_row['num_of_prev_attempts'])} lần). "
                "Cần theo dõi sát tiến độ ngay từ 4 tuần đầu học kỳ."
            )
        if not recs:
            if risk_level == "Low":
                recs.append("✅ TIẾN ĐỘ TỐT: Sinh viên duy trì lịch nộp bài và điểm số ổn định. Tiếp tục theo dõi định kỳ.")
            else:
                recs.append("ℹ️ THEO DÕI SỚM: Rủi ro đến từ yếu tố nền tảng đầu vào hoặc môn học có tỷ lệ trượt cao; nên gửi thông báo nhắc mốc bài kiểm tra đầu kỳ.")
        return recs
