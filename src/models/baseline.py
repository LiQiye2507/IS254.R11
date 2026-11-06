"""
Mô hình Đối chứng (Baseline Models) cho Hệ DSS Cảnh báo sớm Nguy cơ Học tập (IS254.R11)
---------------------------------------------------------------------------------------
Bao gồm:
  - Logistic Regression (tuyến tính, chuẩn hóa z-score, trọng số lớp cân bằng)
  - Random Forest Classifier (phi tuyến trên đặc trưng tích lũy đến tuần T)
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASELINE_CAT_COLS = [
    'gender', 'region', 'highest_education', 'imd_band',
    'age_band', 'disability', 'code_module'
]

BASELINE_NUM_COLS = [
    'num_of_prev_attempts', 'studied_credits',
    'due_count', 'submitted_count', 'missed_count',
    'has_submitted', 'has_active', 'submission_rate',
    'sub_mean_score', 'eff_mean_score', 'weighted_score',
    'late_count', 'fail_count', 'tma_bad_count',
    'last_score', 'score_trend'
]


def build_baseline_pipeline(model_type: str = 'rf', random_state: int = 42) -> Pipeline:
    """
    Tạo sklearn Pipeline gồm tiền xử lý (OneHot + StandardScaler) và bộ phân loại đối chứng.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), BASELINE_CAT_COLS),
            ('num', StandardScaler(), BASELINE_NUM_COLS),
        ],
        remainder='drop',
    )

    if model_type == 'lr':
        clf = LogisticRegression(
            max_iter=600,
            class_weight='balanced',
            C=1.0,
            random_state=random_state,
        )
    elif model_type == 'rf':
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=10,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        raise ValueError(f"model_type không hợp lệ: {model_type}")

    return Pipeline([('prep', preprocessor), ('clf', clf)])


def fit_and_predict_baseline(
    train_snap: pd.DataFrame,
    test_snap: pd.DataFrame,
    model_type: str = 'rf',
    random_state: int = 42,
):
    """
    Huấn luyện mô hình đối chứng tại tuần T trên tập train_snap và trả về xác suất P(at_risk=1) trên test_snap.
    """
    feature_cols = BASELINE_CAT_COLS + BASELINE_NUM_COLS
    X_train = train_snap[feature_cols]
    y_train = train_snap['at_risk'].to_numpy(dtype=np.int32)
    X_test = test_snap[feature_cols]

    pipe = build_baseline_pipeline(model_type=model_type, random_state=random_state)
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]
    return pipe, probs
