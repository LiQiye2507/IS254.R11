"""
Kiểm thử chống rò rỉ thông tin tương lai (Mục 8 DSS_plan.docx).

Ý tưởng chung: thay đổi tùy ý dữ liệu phát sinh SAU tuần T (điểm, nộp trễ, click LMS);
mọi đặc trưng dùng để dự đoán tại tuần T phải giữ nguyên.

Chạy: python -m pytest tests -q
"""

import os
import sys

import numpy as np
import pytest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.api.service import dbn_effective_week
from src.data.temporal_builder import (
    CLICK_COLS,
    DAY_COLS,
    DBN_SLICES,
    build_dbn_discretized_dataset,
    build_weekly_sequence_tensors,
    compute_snapshot_features_at_week,
    load_base_enrollments_and_schedule,
)
from src.models.lstm_model import EarlyWarningLSTMNet

WEEK_T = 8


def _perturb_after_week(enrollments, grid, week_t):
    """Bản sao dữ liệu với mọi thông tin phát sinh sau tuần T bị thay đổi."""
    enr, grd = enrollments.copy(), grid.copy()
    enr[CLICK_COLS[week_t:] + DAY_COLS[week_t:]] = 999.0
    future = grd['submission_week'] > week_t
    grd.loc[future, 'score'] = 100.0 - grd.loc[future, 'score'].fillna(0.0)
    grd.loc[future, 'is_late'] = 1 - grd.loc[future, 'is_late'].fillna(0)
    return enr, grd


@pytest.fixture(scope='module')
def data():
    enrollments, grid = load_base_enrollments_and_schedule()
    return enrollments, grid, *_perturb_after_week(enrollments, grid, WEEK_T)


def test_perturbation_changes_something(data):
    enr, grd, enr2, grd2 = data
    assert not enr.equals(enr2) and not grd.equals(grd2)


def test_snapshot_features_ignore_future(data):
    enr, grd, enr2, grd2 = data
    a = compute_snapshot_features_at_week(enr, grd, WEEK_T)
    b = compute_snapshot_features_at_week(enr2, grd2, WEEK_T)
    assert a.equals(b)


def test_lstm_sequence_ignores_future(data):
    enr, grd, enr2, grd2 = data
    x1, _ = build_weekly_sequence_tensors(enr, grd)
    x2, _ = build_weekly_sequence_tensors(enr2, grd2)
    assert np.array_equal(x1[:, :WEEK_T], x2[:, :WEEK_T])
    assert not np.array_equal(x1[:, WEEK_T:], x2[:, WEEK_T:])


def test_dbn_observed_slices_ignore_future(data):
    enr, grd, enr2, grd2 = data
    d1 = build_dbn_discretized_dataset(enr, grd)
    d2 = build_dbn_discretized_dataset(enr2, grd2)
    for s_name, _, w_end in DBN_SLICES:
        if w_end <= WEEK_T:
            for prefix in ('perf', 'disc', 'click', 'state'):
                col = f'{prefix}_{s_name}'
                assert d1[col].equals(d2[col]), col


def test_dbn_week_mapping_never_uses_future_slice():
    for week_t in range(1, 21):
        eff = dbn_effective_week(week_t)
        assert eff <= week_t
        assert all(w_end <= eff for _, _, w_end in DBN_SLICES if w_end <= eff)


def test_lstm_output_at_t_depends_only_on_past():
    torch.manual_seed(0)
    net = EarlyWarningLSTMNet(seq_dim=18, static_dim=5).eval()
    x_seq, x_stat = torch.randn(4, 20, 18), torch.randn(4, 5)
    x_future = x_seq.clone()
    x_future[:, WEEK_T:] = torch.randn(4, 20 - WEEK_T, 18)
    with torch.no_grad():
        out1, out2 = net(x_seq, x_stat), net(x_future, x_stat)
    assert torch.allclose(out1[:, :WEEK_T], out2[:, :WEEK_T])
    assert not torch.allclose(out1[:, WEEK_T:], out2[:, WEEK_T:])
