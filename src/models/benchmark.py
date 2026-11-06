"""
Quy trình Huấn luyện & Đánh giá Thực nghiệm 5-Fold Cross-Validation cho Hệ DSS (IS254.R11)
------------------------------------------------------------------------------------------
- Phân chia dữ liệu: StratifiedGroupKFold (5 folds) theo `id_student` (không rò rỉ sinh viên).
- Các mốc tuần đánh giá cảnh báo sớm: T in {2, 4, 8, 12, 16, 20} (bao gồm T=2, 8, 16 của Bảng 4).
- Các mô hình so sánh:
    1. Baseline (LR) : Logistic Regression
    2. Baseline (RF) : Random Forest
    3. DBN (5-Tier)  : Dynamic Bayesian Network 5 tầng (pgmpy - HillClimb BIC + VariableElimination)
    4. LSTM (Masked) : Many-to-Many Causal Masked LSTM (PyTorch)
- Lưu trữ mô hình tốt nhất vào `models/` phục vụ FastAPI Backend & Streamlit Dashboard.
- Xuất kết quả chi tiết ra `data/processed/benchmark_folds.csv`, `data/processed/benchmark_summary.csv`
  và báo cáo đối chiếu giả thuyết H1, H2, H3 tại `docs/experimental_results.md`.
"""

import argparse
import json
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.temporal_builder import (
    EVAL_WEEKS,
    build_dbn_discretized_dataset,
    build_weekly_sequence_tensors,
    compute_snapshot_features_at_week,
    load_base_enrollments_and_schedule,
)
from src.models.baseline import fit_and_predict_baseline
from src.models.dbn_model import FiveTierDBNClassifier
from src.models.lstm_model import MaskedLSTMTrainer


def compute_all_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(np.int32)
    return {
        'accuracy': float(accuracy_score(y_true, preds)),
        'precision': float(precision_score(y_true, preds, zero_division=0)),
        'recall': float(recall_score(y_true, preds, zero_division=0)),
        'f1': float(f1_score(y_true, preds, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_true, probs)),
        'auc_pr': float(average_precision_score(y_true, probs)),
    }


def run_benchmark(n_splits: int = 5, max_folds_to_run: int | None = None, save_artifacts: bool = True):
    t_global = time.time()
    print("=" * 78)
    print("🚀 KHỞI ĐỘNG QUY TRÌNH HUẤN LUYỆN & ĐÁNH GIÁ HỆ DSS (DBN vs LSTM vs BASELINE)")
    print("=" * 78)

    # 1. Tải dữ liệu cơ sở và xây dựng các biểu diễn theo thời gian
    t0 = time.time()
    print("\n[1/4] Đang chuẩn bị dữ liệu lịch thi, lát cắt tuần T và chuỗi thời gian 3D...")
    enrollments, grid = load_base_enrollments_and_schedule()
    snapshots_by_week = {
        w: compute_snapshot_features_at_week(enrollments, grid, w)
        for w in EVAL_WEEKS
    }
    dbn_full_df = build_dbn_discretized_dataset(enrollments, grid)
    seq_x, loss_mask = build_weekly_sequence_tensors(enrollments, grid)

    print(f"  + Số lượt đăng ký học phần : {len(enrollments):,} lượt ({enrollments['id_student'].nunique():,} SV duy nhất)")
    print(f"  + Tỷ lệ nhãn At-risk       : {enrollments['at_risk'].mean()*100:.2f}%")
    print(f"  + Kích thước Tensor LSTM   : {seq_x.shape} (N x 20 tuần x 12 đặc trưng)")
    print(f"  + Kích thước bảng DBN      : {dbn_full_df.shape} (5 tầng x 6 lát cắt S1..S6)")
    print(f"  + Thời gian tạo đặc trưng  : {time.time() - t0:.2f}s")

    # 2. Thiết lập StratifiedGroupKFold theo id_student
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_all = enrollments['at_risk'].to_numpy(dtype=np.int32)
    groups = enrollments['id_student'].to_numpy()

    fold_records = []
    saved_sample_explanation = None
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    if save_artifacts:
        os.makedirs(models_dir, exist_ok=True)

    print(f"\n[2/4] Bắt đầu huấn luyện kiểm định chéo {n_splits}-Fold (nhóm theo id_student)...")

    for fold_idx, (tr_idx, te_idx) in enumerate(sgkf.split(enrollments, y_all, groups=groups), start=1):
        if max_folds_to_run is not None and fold_idx > max_folds_to_run:
            break

        t_fold = time.time()
        # Kiểm chứng không rò rỉ sinh viên giữa Train và Test
        overlap = set(groups[tr_idx]).intersection(set(groups[te_idx]))
        assert len(overlap) == 0, f"Phát hiện rò rỉ {len(overlap)} sinh viên ở Fold {fold_idx}!"

        y_test = y_all[te_idx]
        print(f"\n--- Fold {fold_idx}/{n_splits} (Train: {len(tr_idx):,} | Test: {len(te_idx):,} | Overlap SV: 0) ---")

        # ------------------------------------------------------------------
        # A. Huấn luyện Mô hình Mạng Bayesian Động 5 Tầng (DBN - pgmpy)
        # ------------------------------------------------------------------
        t_dbn = time.time()
        dbn_clf = FiveTierDBNClassifier(max_indegree=4, ess=15.0)
        dbn_clf.fit(dbn_full_df.iloc[tr_idx].reset_index(drop=True), eval_weeks=EVAL_WEEKS)
        dbn_test_df = dbn_full_df.iloc[te_idx].reset_index(drop=True)

        for w in EVAL_WEEKS:
            p_dbn = dbn_clf.predict_proba_at_week(dbn_test_df, w)
            th_dbn = dbn_clf.thresholds_by_week.get(w, 0.42)
            m = compute_all_metrics(y_test, p_dbn, threshold=th_dbn)
            fold_records.append({'fold': fold_idx, 'week': w, 'model': 'DBN', **m})
        print(f"  + [DBN 5-Tier]  Hoàn tất ({time.time() - t_dbn:.2f}s) | Số cạnh học được: {len(dbn_clf.learned_edges)}")

        # ------------------------------------------------------------------
        # B. Huấn luyện Mô hình Chuỗi Masked Many-to-Many LSTM (PyTorch)
        # ------------------------------------------------------------------
        t_lstm = time.time()
        lstm_trainer = MaskedLSTMTrainer(
            hidden_dim=64,
            num_layers=2,
            dropout=0.25,
            lr=1.5e-3,
            batch_size=256,
            max_epochs=26,
            patience=5,
            random_state=42 + fold_idx,
        )
        lstm_trainer.fit(
            enrollments.iloc[tr_idx].reset_index(drop=True),
            seq_x[tr_idx],
            loss_mask[tr_idx],
            eval_weeks=EVAL_WEEKS,
        )
        probs_lstm_all = lstm_trainer.predict_proba_all_weeks(
            enrollments.iloc[te_idx].reset_index(drop=True),
            seq_x[te_idx],
        )
        for w in EVAL_WEEKS:
            p_lstm = probs_lstm_all[:, w - 1]
            th_lstm = lstm_trainer.thresholds_by_week.get(w, 0.42)
            m = compute_all_metrics(y_test, p_lstm, threshold=th_lstm)
            fold_records.append({'fold': fold_idx, 'week': w, 'model': 'LSTM', **m})
        print(f"  + [Masked LSTM] Hoàn tất ({time.time() - t_lstm:.2f}s)")

        # ------------------------------------------------------------------
        # C. Huấn luyện Mô hình Đối chứng (Baseline RF & LR) theo từng tuần T
        # ------------------------------------------------------------------
        t_base = time.time()
        rf_models_by_w = {}
        for w in EVAL_WEEKS:
            snap_w = snapshots_by_week[w]
            tr_snap = snap_w.iloc[tr_idx].reset_index(drop=True)
            te_snap = snap_w.iloc[te_idx].reset_index(drop=True)

            rf_pipe, p_rf = fit_and_predict_baseline(tr_snap, te_snap, model_type='rf', random_state=42 + fold_idx)
            m_rf = compute_all_metrics(y_test, p_rf, threshold=0.5)
            fold_records.append({'fold': fold_idx, 'week': w, 'model': 'Baseline_RF', **m_rf})

            _, p_lr = fit_and_predict_baseline(tr_snap, te_snap, model_type='lr', random_state=42 + fold_idx)
            m_lr = compute_all_metrics(y_test, p_lr, threshold=0.5)
            fold_records.append({'fold': fold_idx, 'week': w, 'model': 'Baseline_LR', **m_lr})

            if fold_idx == 1:
                rf_models_by_w[w] = rf_pipe

        print(f"  + [Baseline]    Hoàn tất RF & LR trên 6 mốc tuần ({time.time() - t_base:.2f}s)")
        print(f"  => Tổng thời gian Fold {fold_idx}: {time.time() - t_fold:.2f}s")

        # Lưu model artifacts từ Fold 1 và trích xuất ví dụ giải thích nhân quả DBN (H3)
        if fold_idx == 1 and save_artifacts:
            joblib.dump(dbn_clf, os.path.join(models_dir, 'dbn_5tier.joblib'), compress=3)
            joblib.dump(rf_models_by_w, os.path.join(models_dir, 'baseline_rf_by_week.joblib'), compress=3)
            torch.save(
                {
                    'model_state_dict': lstm_trainer.model.state_dict(),
                    'static_prep': lstm_trainer.static_prep,
                    'seq_mean': lstm_trainer.seq_mean,
                    'seq_std': lstm_trainer.seq_std,
                    'pos_weight_val': lstm_trainer.pos_weight_val,
                    'thresholds_by_week': lstm_trainer.thresholds_by_week,
                    'hidden_dim': lstm_trainer.hidden_dim,
                    'num_layers': lstm_trainer.num_layers,
                    'dropout': lstm_trainer.dropout,
                },
                os.path.join(models_dir, 'lstm_masked.pt'),
            )

            # Chọn 1 sinh viên có nguy cơ cao thực tế ở T=8 để minh họa giải thích nhân quả DBN (H3)
            p8 = dbn_clf.predict_proba_at_week(dbn_test_df, 8)
            high_risk_candidates = np.where((y_test == 1) & (p8 >= 0.70))[0]
            sample_i = int(high_risk_candidates[0]) if len(high_risk_candidates) > 0 else 0
            enr_sample = enrollments.iloc[te_idx[sample_i]]
            exp_dict = dbn_clf.explain_student(dbn_test_df.iloc[sample_i], week_t=8, top_k=5)
            exp_dict['id_student'] = int(enr_sample['id_student'])
            exp_dict['code_module'] = str(enr_sample['code_module'])
            exp_dict['code_presentation'] = str(enr_sample['code_presentation'])
            exp_dict['actual_final_result'] = str(enr_sample['final_result'])
            saved_sample_explanation = exp_dict
            with open(os.path.join(models_dir, 'sample_dbn_explanation.json'), 'w', encoding='utf-8') as f:
                json.dump(saved_sample_explanation, f, ensure_ascii=False, indent=2)

    # 3. Tổng hợp kết quả trung bình & độ lệch chuẩn qua các fold
    df_folds = pd.DataFrame(fold_records)
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr']
    summary_rows = []
    for (w, m_name), grp in df_folds.groupby(['week', 'model']):
        row = {'week': int(w), 'model': m_name}
        for met in metric_names:
            row[f'{met}_mean'] = float(grp[met].mean())
            row[f'{met}_std'] = float(grp[met].std(ddof=1)) if len(grp) > 1 else 0.0
        summary_rows.append(row)

    model_order = {'DBN': 0, 'LSTM': 1, 'Baseline_RF': 2, 'Baseline_LR': 3}
    df_summary = pd.DataFrame(summary_rows)
    df_summary['_ord'] = df_summary['model'].map(model_order)
    df_summary = df_summary.sort_values(['week', '_ord']).drop(columns=['_ord']).reset_index(drop=True)

    if save_artifacts:
        folds_csv = os.path.join(PROJECT_ROOT, 'data', 'processed', 'benchmark_folds.csv')
        sum_csv = os.path.join(PROJECT_ROOT, 'data', 'processed', 'benchmark_summary.csv')
        df_folds.to_csv(folds_csv, index=False)
        df_summary.to_csv(sum_csv, index=False)
        _write_markdown_report(df_summary, saved_sample_explanation, n_folds_actual=df_folds['fold'].nunique())

    print("\n" + "=" * 78)
    print("📊 BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM (TRUNG BÌNH ± ĐỘ LỆCH CHUẨN)")
    print("=" * 78)
    for w in EVAL_WEEKS:
        print(f"\n>>> MỐC TUẦN T = {w}:")
        sub = df_summary[df_summary['week'] == w]
        for _, r in sub.iterrows():
            print(
                f"  {r['model']:<13} | Acc: {r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f} "
                f"| Prec: {r['precision_mean']:.4f} | Rec: {r['recall_mean']:.4f} "
                f"| F1: {r['f1_mean']:.4f}±{r['f1_std']:.4f} | AUC-ROC: {r['auc_roc_mean']:.4f}±{r['auc_roc_std']:.4f}"
            )

    print("\n" + "=" * 78)
    print(f"✅ HOÀN TẤT TOÀN BỘ THỰC NGHIỆM TRONG {time.time() - t_global:.2f} GIÂY!")
    print("=" * 78)
    return df_folds, df_summary


def _write_markdown_report(df_summary: pd.DataFrame, sample_exp: dict | None, n_folds_actual: int):
    report_path = os.path.join(PROJECT_ROOT, 'docs', 'experimental_results.md')
    lines = [
        "# Kết Quả Thực Nghiệm & Đối Chiếu Giả Thuyết (IS254.R11)",
        "",
        f"> **Thiết lập đánh giá:** Kiểm định chéo `{n_folds_actual}-Fold StratifiedGroupKFold` chia theo `id_student` trên **25.814 lượt đăng ký** (`23.345` sinh viên duy nhất, tỷ lệ `At-risk = 40.41%`). Không có bất kỳ sinh viên nào xuất hiện đồng thời ở tập Train và Test.",
        "",
        "---",
        "",
        "## 1. Bảng 4 (Điền số liệu chính thức cho Đề cương `DSS_plan.docx`: `T = 2, 8, 16`)",
        "",
        "| Tuần | Mô hình | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUC-PR |",
        "| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: |",
    ]

    display_name = {
        'DBN': 'DBN (5-Tier)',
        'LSTM': 'LSTM (Masked)',
        'Baseline_RF': 'Đối chứng (RF)',
        'Baseline_LR': 'Đối chứng (LR)',
    }

    for w in [2, 8, 16]:
        sub = df_summary[df_summary['week'] == w]
        for _, r in sub.iterrows():
            lines.append(
                f"| **T = {w}** | {display_name.get(r['model'], r['model'])} "
                f"| {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f} "
                f"| {r['precision_mean']:.4f} ± {r['precision_std']:.4f} "
                f"| {r['recall_mean']:.4f} ± {r['recall_std']:.4f} "
                f"| {r['f1_mean']:.4f} ± {r['f1_std']:.4f} "
                f"| {r['auc_roc_mean']:.4f} ± {r['auc_roc_std']:.4f} "
                f"| {r['auc_pr_mean']:.4f} ± {r['auc_pr_std']:.4f} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Toàn bộ đường cong dự đoán sớm trên 6 mốc tuần (`T ∈ {2, 4, 8, 12, 16, 20}`)",
        "",
        "| Tuần | Mô hình | Recall (At-risk) | F1-Score | AUC-ROC | AUC-PR |",
        "| :-: | :--- | :-: | :-: | :-: | :-: |",
    ])

    for w in EVAL_WEEKS:
        sub = df_summary[df_summary['week'] == w]
        for _, r in sub.iterrows():
            lines.append(
                f"| T = {w} | {display_name.get(r['model'], r['model'])} "
                f"| {r['recall_mean']:.4f} ± {r['recall_std']:.4f} "
                f"| {r['f1_mean']:.4f} ± {r['f1_std']:.4f} "
                f"| {r['auc_roc_mean']:.4f} ± {r['auc_roc_std']:.4f} "
                f"| {r['auc_pr_mean']:.4f} ± {r['auc_pr_std']:.4f} |"
            )

    if sample_exp is not None:
        lines.extend([
            "",
            "---",
            "",
            "## 3. Minh họa khả năng giải thích nhân quả của DBN (Kiểm chứng Giả thuyết H3)",
            "",
            f"* **Sinh viên kiểm thử:** `id_student = {sample_exp['id_student']}` | Môn: `{sample_exp['code_module']}` (`{sample_exp['code_presentation']}`)",
            f"* **Thời điểm dự báo:** Tuần `T = {sample_exp['week_t']}`",
            f"* **Xác suất rủi ro DBN dự đoán:** `P(At-risk) = {sample_exp['prob_at_risk']*100:.2f}%` → Phân loại mức: **`{sample_exp['risk_level']}`**",
            f"* **Kết quả thực tế cuối kỳ:** `{sample_exp['actual_final_result']}`",
            "",
            "| Thứ tự | Yếu tố bằng chứng quan sát (`Factor`) | Giá trị của SV (`Value`) | Độ đóng góp rủi ro (`Delta P`) |",
            "| :-: | :--- | :--- | :-: |",
        ])
        for i, item in enumerate(sample_exp['top_factors'], start=1):
            sign = "+" if item['delta_prob'] >= 0 else ""
            lines.append(
                f"| {i} | `{item['factor']}` | `{item['value']}` | **{sign}{item['delta_prob']*100:.2f}%** |"
            )

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', type=int, default=5)
    parser.add_argument('--max-folds', type=int, default=None)
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()

    run_benchmark(
        n_splits=args.splits,
        max_folds_to_run=args.max_folds,
        save_artifacts=not args.no_save,
    )
