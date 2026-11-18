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
    5. Hybrid        : a * P(DBN) + (1 - a) * P(LSTM), trọng số a chọn trên Validation theo từng tuần
- Mỗi fold tách 15% sinh viên của tập Train làm Validation chung: mọi mô hình học trên 85% còn lại và
  chọn ngưỡng phân loại (tối đa F1) theo cùng một cách trên Validation.
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
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

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


MODEL_ORDER = ['DBN', 'LSTM', 'Hybrid', 'Baseline_RF', 'Baseline_LR']
DISPLAY_NAME = {
    'DBN': 'DBN (5-Tier)',
    'LSTM': 'LSTM (Masked)',
    'Hybrid': 'Hybrid (DBN + LSTM)',
    'Baseline_RF': 'Đối chứng (RF)',
    'Baseline_LR': 'Đối chứng (LR)',
}
HYBRID_DBN_WEIGHTS = np.round(np.linspace(0.0, 1.0, 11), 2)
THRESHOLD_GRID = np.round(np.arange(0.20, 0.801, 0.01), 2)


def tune_f1_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Ngưỡng phân loại tối đa F1 trên tập Validation (dùng chung cho mọi mô hình)."""
    return float(max(THRESHOLD_GRID, key=lambda th: f1_score(y_val, (p_val >= th).astype(int), zero_division=0)))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """ECE: trung bình có trọng số |tỷ lệ At-risk thực tế − xác suất dự đoán| trên 10 khoảng xác suất đều nhau."""
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    return float(sum(
        abs(y_true[bins == b].mean() - probs[bins == b].mean()) * (bins == b).mean()
        for b in range(n_bins) if (bins == b).any()
    ))


def compute_all_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(np.int32)
    return {
        'accuracy': float(accuracy_score(y_true, preds)),
        'precision': float(precision_score(y_true, preds, zero_division=0)),
        'recall': float(recall_score(y_true, preds, zero_division=0)),
        'f1': float(f1_score(y_true, preds, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_true, probs)),
        'auc_pr': float(average_precision_score(y_true, probs)),
        'brier': float(brier_score_loss(y_true, probs)),
        'ece': expected_calibration_error(y_true, probs),
    }


def _prepare_data() -> dict:
    """Tải dữ liệu và xây dựng sẵn 3 dạng biểu diễn theo thời gian (bảng tuần T, bảng DBN, tensor LSTM)."""
    t0 = time.time()
    print("\n[1/4] Đang chuẩn bị dữ liệu lịch thi, lát cắt tuần T và chuỗi thời gian 3D...")
    enrollments, grid = load_base_enrollments_and_schedule()
    seq_x, loss_mask = build_weekly_sequence_tensors(enrollments, grid)
    data = {
        'enrollments': enrollments,
        'snapshots_by_week': {w: compute_snapshot_features_at_week(enrollments, grid, w) for w in EVAL_WEEKS},
        'dbn_full_df': build_dbn_discretized_dataset(enrollments, grid),
        'seq_x': seq_x,
        'loss_mask': loss_mask,
        'y_all': enrollments['at_risk'].to_numpy(dtype=np.int32),
        'groups': enrollments['id_student'].to_numpy(),
    }
    print(f"  + Số lượt đăng ký học phần : {len(enrollments):,} lượt ({enrollments['id_student'].nunique():,} SV duy nhất)")
    print(f"  + Tỷ lệ nhãn At-risk       : {enrollments['at_risk'].mean()*100:.2f}%")
    print(f"  + Kích thước Tensor LSTM   : {seq_x.shape} (N x 20 tuần x {seq_x.shape[2]} đặc trưng)")
    print(f"  + Kích thước bảng DBN      : {data['dbn_full_df'].shape} (5 tầng x 6 lát cắt S1..S6)")
    print(f"  + Thời gian tạo đặc trưng  : {time.time() - t0:.2f}s")
    return data


def train_and_evaluate(data: dict, tr_idx: np.ndarray, te_idx: np.ndarray, seed: int):
    """
    Huấn luyện 5 mô hình trên tr_idx và đánh giá trên te_idx tại các mốc EVAL_WEEKS.
    tr_idx được tách theo sinh viên: 85% để học, 15% làm Validation chung cho mọi mô hình
    (early stopping LSTM, chọn ngưỡng phân loại, chọn trọng số Hybrid).
    Trả về (danh sách bản ghi chỉ số, trọng số Hybrid theo tuần, các mô hình đã huấn luyện).
    """
    enrollments, snapshots_by_week = data['enrollments'], data['snapshots_by_week']
    dbn_full_df, seq_x, loss_mask = data['dbn_full_df'], data['seq_x'], data['loss_mask']
    y_all, groups = data['y_all'], data['groups']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    fit_rel, val_rel = next(gss.split(tr_idx, groups=groups[tr_idx]))
    fit_idx, val_idx = tr_idx[fit_rel], tr_idx[val_rel]
    assert not set(groups[fit_idx]) & set(groups[val_idx])
    y_val, y_test = y_all[val_idx], y_all[te_idx]
    val_probs, test_probs = {}, {}  # model -> {week: xác suất}
    print(f"  + Fit: {len(fit_idx):,} | Val: {len(val_idx):,} | Test: {len(te_idx):,}")

    # A. Mạng Bayesian Động 5 Tầng (DBN - pgmpy)
    t_dbn = time.time()
    dbn_clf = FiveTierDBNClassifier(max_indegree=4, ess=15.0)
    dbn_clf.fit(dbn_full_df.iloc[fit_idx].reset_index(drop=True))
    dbn_val_df = dbn_full_df.iloc[val_idx].reset_index(drop=True)
    dbn_test_df = dbn_full_df.iloc[te_idx].reset_index(drop=True)
    val_probs['DBN'] = {w: dbn_clf.predict_proba_at_week(dbn_val_df, w) for w in EVAL_WEEKS}
    test_probs['DBN'] = {w: dbn_clf.predict_proba_at_week(dbn_test_df, w) for w in EVAL_WEEKS}
    print(f"  + [DBN 5-Tier]  Hoàn tất ({time.time() - t_dbn:.2f}s) | Số cạnh học được: {len(dbn_clf.learned_edges)}")

    # B. Masked Many-to-Many LSTM (PyTorch)
    t_lstm = time.time()
    lstm_trainer = MaskedLSTMTrainer(
        hidden_dim=64, num_layers=2, dropout=0.25, lr=1.5e-3,
        batch_size=256, max_epochs=26, patience=5, random_state=seed,
    )
    enr_val = enrollments.iloc[val_idx].reset_index(drop=True)
    lstm_trainer.fit(
        enrollments.iloc[fit_idx].reset_index(drop=True), seq_x[fit_idx], loss_mask[fit_idx],
        enr_val, seq_x[val_idx], eval_weeks=EVAL_WEEKS,
    )
    p_val_all = lstm_trainer.predict_proba_all_weeks(enr_val, seq_x[val_idx])
    p_test_all = lstm_trainer.predict_proba_all_weeks(enrollments.iloc[te_idx].reset_index(drop=True), seq_x[te_idx])
    val_probs['LSTM'] = {w: p_val_all[:, w - 1] for w in EVAL_WEEKS}
    test_probs['LSTM'] = {w: p_test_all[:, w - 1] for w in EVAL_WEEKS}
    print(f"  + [Masked LSTM] Hoàn tất ({time.time() - t_lstm:.2f}s)")

    # C. Mô hình đối chứng (RF & LR) huấn luyện riêng cho từng tuần T
    t_base = time.time()
    rf_models_by_w = {}
    for name, model_type in [('Baseline_RF', 'rf'), ('Baseline_LR', 'lr')]:
        val_probs[name], test_probs[name] = {}, {}
        for w in EVAL_WEEKS:
            snap_w = snapshots_by_week[w]
            pipe, test_probs[name][w] = fit_and_predict_baseline(
                snap_w.iloc[fit_idx].reset_index(drop=True), snap_w.iloc[te_idx].reset_index(drop=True),
                model_type=model_type, random_state=seed,
            )
            val_probs[name][w] = pipe.predict_proba(snap_w.iloc[val_idx].reset_index(drop=True))[:, 1]
            if model_type == 'rf':
                rf_models_by_w[w] = pipe
    print(f"  + [Baseline]    Hoàn tất RF & LR trên 6 mốc tuần ({time.time() - t_base:.2f}s)")

    # D. Hybrid DSS (RQ3): P = a * P_DBN + (1 - a) * P_LSTM, trọng số a chọn theo AUC-ROC trên Validation
    weights = {}
    val_probs['Hybrid'], test_probs['Hybrid'] = {}, {}
    for w in EVAL_WEEKS:
        a = max(
            HYBRID_DBN_WEIGHTS,
            key=lambda x: roc_auc_score(y_val, x * val_probs['DBN'][w] + (1 - x) * val_probs['LSTM'][w]),
        )
        weights[w] = float(a)
        val_probs['Hybrid'][w] = a * val_probs['DBN'][w] + (1 - a) * val_probs['LSTM'][w]
        test_probs['Hybrid'][w] = a * test_probs['DBN'][w] + (1 - a) * test_probs['LSTM'][w]
    print(f"  + [Hybrid]      Trọng số DBN theo tuần: {weights}")

    # Mọi mô hình: chọn ngưỡng F1 trên Validation theo cùng một cách, rồi đánh giá trên Test
    records = []
    for name in MODEL_ORDER:
        for w in EVAL_WEEKS:
            th = tune_f1_threshold(y_val, val_probs[name][w])
            records.append({'week': w, 'model': name, 'threshold': th,
                            **compute_all_metrics(y_test, test_probs[name][w], threshold=th)})

    trained = {'dbn': dbn_clf, 'lstm': lstm_trainer, 'rf_by_week': rf_models_by_w, 'dbn_test_df': dbn_test_df}
    return records, weights, trained


def run_benchmark(n_splits: int = 5, max_folds_to_run: int | None = None, save_artifacts: bool = True):
    t_global = time.time()
    print("=" * 78)
    print("🚀 KHỞI ĐỘNG QUY TRÌNH HUẤN LUYỆN & ĐÁNH GIÁ HỆ DSS (DBN vs LSTM vs BASELINE)")
    print("=" * 78)

    # 1. Tải dữ liệu cơ sở và xây dựng các biểu diễn theo thời gian
    data = _prepare_data()
    enrollments, y_all, groups = data['enrollments'], data['y_all'], data['groups']

    # 2. Thiết lập StratifiedGroupKFold theo id_student
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_records = []
    hybrid_weights = {}  # fold -> {week: trọng số DBN}
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
        records, hybrid_weights[fold_idx], trained = train_and_evaluate(data, tr_idx, te_idx, seed=42 + fold_idx)
        fold_records.extend({'fold': fold_idx, **r} for r in records)
        dbn_clf, lstm_trainer, rf_models_by_w, dbn_test_df = (
            trained['dbn'], trained['lstm'], trained['rf_by_week'], trained['dbn_test_df']
        )
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
                    'hidden_dim': lstm_trainer.hidden_dim,
                    'num_layers': lstm_trainer.num_layers,
                    'dropout': lstm_trainer.dropout,
                },
                os.path.join(models_dir, 'lstm_masked.pt'),
            )
            with open(os.path.join(models_dir, 'hybrid_weights.json'), 'w', encoding='utf-8') as f:
                json.dump(hybrid_weights[1], f, indent=2)

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
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr', 'brier', 'ece', 'threshold']
    summary_rows = []
    for (w, m_name), grp in df_folds.groupby(['week', 'model']):
        row = {'week': int(w), 'model': m_name}
        for met in metric_names:
            row[f'{met}_mean'] = float(grp[met].mean())
            row[f'{met}_std'] = float(grp[met].std(ddof=1)) if len(grp) > 1 else 0.0
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary['_ord'] = df_summary['model'].map(MODEL_ORDER.index)
    df_summary = df_summary.sort_values(['week', '_ord']).drop(columns=['_ord']).reset_index(drop=True)

    if save_artifacts:
        folds_csv = os.path.join(PROJECT_ROOT, 'data', 'processed', 'benchmark_folds.csv')
        sum_csv = os.path.join(PROJECT_ROOT, 'data', 'processed', 'benchmark_summary.csv')
        df_folds.to_csv(folds_csv, index=False)
        df_summary.to_csv(sum_csv, index=False)
        with open(os.path.join(PROJECT_ROOT, 'data', 'processed', 'hybrid_weights_by_fold.json'), 'w', encoding='utf-8') as f:
            json.dump(hybrid_weights, f, indent=2)
        _write_markdown_report(
            df_summary, df_folds, saved_sample_explanation, n_folds_actual=df_folds['fold'].nunique(),
            enrollments=enrollments, hybrid_weights=hybrid_weights,
        )

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


def _write_markdown_report(df_summary: pd.DataFrame, df_folds: pd.DataFrame, sample_exp: dict | None, n_folds_actual: int, enrollments: pd.DataFrame, hybrid_weights: dict):
    report_path = os.path.join(PROJECT_ROOT, 'docs', 'experimental_results.md')
    lines = [
        "# Kết Quả Thực Nghiệm & Đối Chiếu Giả Thuyết (IS254.R11)",
        "",
        f"> **Thiết lập đánh giá:** Kiểm định chéo `{n_folds_actual}-Fold StratifiedGroupKFold` chia theo `id_student` trên **{len(enrollments):,} lượt đăng ký** (`{enrollments['id_student'].nunique():,}` sinh viên duy nhất; tỷ lệ `At-risk = {enrollments['at_risk'].mean() * 100:.2f}%`). Không có bất kỳ sinh viên nào xuất hiện đồng thời ở tập Train và Test.".replace(',', '.'),
        "",
        "---",
        "",
        "## 1. Bảng 4 (Điền số liệu chính thức cho Đề cương `DSS_plan.docx`: `T = 2, 8, 16`)",
        "",
        "| Tuần | Mô hình | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUC-PR |",
        "| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: |",
    ]

    display_name = DISPLAY_NAME

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

    # So sánh cặp trên cùng fold: loại bỏ biến động do cách chia dữ liệu khi so hai mô hình
    lines.extend([
        "",
        "---",
        "",
        "## 4. So sánh cặp theo từng fold (A − B)",
        "",
        "Mỗi ô: hiệu số trung bình ± độ lệch chuẩn qua các fold, trong ngoặc là số fold A > B. "
        "Phân tích giả thuyết H1, H2, H3 dựa trên bảng này: xem [hypothesis_analysis.md](hypothesis_analysis.md).",
        "",
        "| Tuần | A − B | ΔAUC-ROC | ΔF1 | ΔRecall |",
        "| :-: | :--- | :-: | :-: | :-: |",
    ])
    piv = df_folds.pivot_table(index=['week', 'fold'], columns='model', values=['auc_roc', 'f1', 'recall'])
    for w in EVAL_WEEKS:
        for a, b in [('DBN', 'LSTM'), ('Hybrid', 'LSTM'), ('LSTM', 'Baseline_RF'), ('LSTM', 'Baseline_LR')]:
            cells = []
            for met in ['auc_roc', 'f1', 'recall']:
                d = (piv[met][a] - piv[met][b]).xs(w, level='week')
                cells.append(f"{d.mean():+.4f} ± {d.std(ddof=1):.4f} ({int((d > 0).sum())}/{len(d)})")
            lines.append(f"| T = {w} | {display_name[a]} − {display_name[b]} | " + " | ".join(cells) + " |")

    # Hybrid: trọng số DBN chọn trên Validation của từng fold
    folds = sorted(hybrid_weights)
    lines.extend([
        "",
        "---",
        "",
        "## 5. Trọng số Hybrid chọn trên Validation (RQ3)",
        "",
        "P(Hybrid) = a · P(DBN) + (1 − a) · P(LSTM); a ∈ {0, 0.1, …, 1} chọn theo AUC-ROC trên tập Validation "
        "(15% sinh viên của tập Train) của từng fold.",
        "",
        "| Tuần | " + " | ".join(f"Fold {k}" for k in folds) + " |",
        "| :-: | " + " | ".join(":-:" for _ in folds) + " |",
    ])
    for w in EVAL_WEEKS:
        lines.append(f"| T = {w} | " + " | ".join(f"{hybrid_weights[k][w]:.1f}" for k in folds) + " |")

    # Hiệu chỉnh xác suất: ngưỡng phân tầng Low/Medium/High dùng trực tiếp xác suất nên cần xác suất đáng tin
    lines.extend([
        "",
        "---",
        "",
        "## 6. Hiệu chỉnh xác suất (Calibration) và ngưỡng phân loại",
        "",
        "Brier score và ECE (10 khoảng xác suất): càng thấp càng tốt. "
        "Ngưỡng: ngưỡng tối đa F1 chọn trên Validation, dùng chung một cách cho mọi mô hình.",
        "",
        "| Tuần | Mô hình | Brier | ECE | Ngưỡng |",
        "| :-: | :--- | :-: | :-: | :-: |",
    ])
    for w in EVAL_WEEKS:
        for _, r in df_summary[df_summary['week'] == w].iterrows():
            lines.append(
                f"| T = {w} | {display_name.get(r['model'], r['model'])} "
                f"| {r['brier_mean']:.4f} ± {r['brier_std']:.4f} "
                f"| {r['ece_mean']:.4f} ± {r['ece_std']:.4f} "
                f"| {r['threshold_mean']:.2f} ± {r['threshold_std']:.2f} |"
            )

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


def run_temporal_holdout():
    """
    Kiểm tra tổng quát theo thời gian (Mục 4 DSS_plan.docx): học trên các kỳ 2013, đánh giá trên các kỳ 2014.
    - Môn chỉ có ở 2014 (CCC) bị loại khỏi tập kiểm tra vì mô hình chưa từng thấy môn này.
    - Sinh viên học lại có mặt ở cả hai năm được giữ: dữ liệu kỳ trước của họ là thông tin hợp lệ khi triển khai.
    Kết quả: data/processed/temporal_holdout.csv và docs/temporal_holdout.md (đối chiếu với 5-fold).
    """
    print("=" * 78)
    print("🕒 KIỂM TRA TỔNG QUÁT THEO THỜI GIAN: HỌC TRÊN 2013 → ĐÁNH GIÁ TRÊN 2014")
    print("=" * 78)
    data = _prepare_data()
    enr = data['enrollments']
    is_2013 = enr['code_presentation'].str.startswith('2013').to_numpy()
    tr_idx = np.where(is_2013)[0]
    te_idx = np.where(~is_2013 & enr['code_module'].isin(enr['code_module'].iloc[tr_idx].unique()).to_numpy())[0]
    n_unseen = int((~is_2013).sum() - len(te_idx))
    n_shared = len(set(data['groups'][tr_idx]) & set(data['groups'][te_idx]))
    print(f"  + Train 2013: {len(tr_idx):,} | Test 2014: {len(te_idx):,} (bỏ {n_unseen:,} lượt môn chưa có ở 2013)")

    records, weights, _ = train_and_evaluate(data, tr_idx, te_idx, seed=42)
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'temporal_holdout.csv'), index=False)

    cv = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'benchmark_summary.csv'))
    cv_auc = cv.set_index(['week', 'model'])['auc_roc_mean']
    vn = lambda n: f"{n:,}".replace(',', '.')  # định dạng số kiểu Việt Nam: 12.555
    lines = [
        "# Kiểm tra tổng quát theo thời gian (học trên 2013 → đánh giá trên 2014)",
        "",
        f"> **Thiết lập:** học trên **{vn(len(tr_idx))}** lượt đăng ký của các kỳ 2013B, 2013J "
        f"(85% để học, 15% sinh viên làm Validation chọn ngưỡng và trọng số Hybrid); đánh giá trên "
        f"**{vn(len(te_idx))}** lượt của các kỳ 2014B, 2014J. Loại {vn(n_unseen)} lượt của môn chỉ có ở 2014 (CCC). "
        f"{vn(n_shared)} sinh viên có mặt ở cả hai năm (học lại) được giữ lại.",
        "",
        "Cột *5-fold* là AUC-ROC trung bình của kiểm định chéo 5-fold (dữ liệu trộn mọi kỳ), "
        "Δ = (2013 → 2014) − (5-fold).",
        "",
        "| Tuần | Mô hình | AUC-ROC (2013 → 2014) | AUC-ROC (5-fold) | Δ | F1 | Recall | Brier | ECE |",
        "| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
    ]
    for _, r in df.iterrows():
        ref = cv_auc.get((r['week'], r['model']), np.nan)
        lines.append(
            f"| T = {r['week']} | {DISPLAY_NAME[r['model']]} | {r['auc_roc']:.4f} | {ref:.4f} | {r['auc_roc'] - ref:+.4f} "
            f"| {r['f1']:.4f} | {r['recall']:.4f} | {r['brier']:.4f} | {r['ece']:.4f} |"
        )
    lines += ["", "Trọng số DBN trong Hybrid chọn trên Validation: "
              + ", ".join(f"T = {w}: {a:.1f}" for w, a in weights.items()) + "."]
    with open(os.path.join(PROJECT_ROOT, 'docs', 'temporal_holdout.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(df[['week', 'model', 'auc_roc', 'f1', 'recall', 'brier', 'ece']].round(4).to_string(index=False))
    return df


if __name__ == '__main__':
    # HillClimbSearch (pgmpy) duyệt tập hợp chuỗi nên cấu trúc DBN phụ thuộc PYTHONHASHSEED;
    # cố định seed (chỉ đặt được trước khi Python khởi động) để kết quả thực nghiệm tái lập được
    if os.environ.get('PYTHONHASHSEED') != '0':
        import subprocess
        sys.exit(subprocess.run([sys.executable] + sys.argv, env={**os.environ, 'PYTHONHASHSEED': '0'}).returncode)

    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', type=int, default=5)
    parser.add_argument('--max-folds', type=int, default=None)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--temporal', action='store_true', help='Học trên 2013, đánh giá trên 2014')
    args = parser.parse_args()

    if args.temporal:
        run_temporal_holdout()
        sys.exit(0)

    run_benchmark(
        n_splits=args.splits,
        max_folds_to_run=args.max_folds,
        save_artifacts=not args.no_save,
    )
