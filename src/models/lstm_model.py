"""
Mô hình Chuỗi Thời Gian Masked Many-to-Many LSTM (PyTorch) cho Hệ DSS (IS254.R11)
---------------------------------------------------------------------------------
Đặc điểm thiết kế bám sát Mục 3.2 & 4 của Đề cương (DSS_plan.docx):
- Đầu vào 3D: (batch_size, timesteps=20, seq_features=12) kết hợp đặc trưng nền (Static).
- Kiến trúc Many-to-Many một chiều nhân quả (Unidirectional Causal LSTM):
  Ẩn trạng thái h_t tại tuần t CHỈ phụ thuộc dữ liệu từ tuần 1..t (Zero Data Leakage).
- Masking các tuần sau khi sinh viên đã hủy môn (loss_mask) trong hàm mất mát.
- Sử dụng trọng số lớp (pos_weight), Early Stopping trên tập Validation (chia theo id_student),
  và hiệu chỉnh xác suất (prior-corrected sigmoid) để P(at_risk=1) bám chuẩn [0, 1] cho Dashboard.
"""

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

STATIC_CAT_COLS = [
    'gender', 'region', 'highest_education', 'imd_band',
    'age_band', 'disability', 'code_module'
]
STATIC_NUM_COLS = ['num_of_prev_attempts', 'studied_credits']


class EarlyWarningLSTMNet(nn.Module):
    def __init__(self, seq_dim: int, static_dim: int, static_emb_dim: int = 32, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, static_emb_dim),
            nn.LayerNorm(static_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(
            input_size=seq_dim + static_emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # Bắt buộc 1 chiều để không rò rỉ thông tin tương lai
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + static_emb_dim, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        """
        x_seq   : (B, T, seq_dim)
        x_static: (B, static_dim)
        returns : logits (B, T)
        """
        _, steps, _ = x_seq.shape
        e_stat = self.static_proj(x_static)                      # (B, E)
        e_stat_rep = e_stat.unsqueeze(1).expand(-1, steps, -1)   # (B, T, E)
        lstm_in = torch.cat([x_seq, e_stat_rep], dim=-1)         # (B, T, seq_dim + E)
        lstm_out, _ = self.lstm(lstm_in)                         # (B, T, H)
        fused = torch.cat([lstm_out, e_stat_rep], dim=-1)        # (B, T, H + E)
        logits = self.head(fused).squeeze(-1)                    # (B, T)
        return logits


class MaskedLSTMTrainer:
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.25,
        lr: float = 1.5e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 28,
        patience: int = 5,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.static_prep = None
        self.seq_mean = None
        self.seq_std = None
        self.model = None
        self.pos_weight_val = 1.0
        self.thresholds_by_week = {}

    def _set_seed(self):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def fit(
        self,
        enr_train: pd.DataFrame,
        seq_train: np.ndarray,
        mask_train: np.ndarray,
        eval_weeks=(2, 4, 8, 12, 16, 20),
    ):
        self._set_seed()

        # Tách 15% sinh viên từ tập Train làm Validation (theo id_student - chống rò rỉ)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=self.random_state)
        sub_tr_idx, val_idx = next(gss.split(enr_train, enr_train['at_risk'], groups=enr_train['id_student']))

        enr_sub = enr_train.iloc[sub_tr_idx].reset_index(drop=True)
        enr_val = enr_train.iloc[val_idx].reset_index(drop=True)

        # Chuẩn hóa đặc trưng nền (chỉ fit trên sub_train)
        self.static_prep = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), STATIC_CAT_COLS),
                ('num', StandardScaler(), STATIC_NUM_COLS),
            ],
            remainder='drop',
        )
        X_stat_tr = self.static_prep.fit_transform(enr_sub[STATIC_CAT_COLS + STATIC_NUM_COLS]).astype(np.float32)
        X_stat_val = self.static_prep.transform(enr_val[STATIC_CAT_COLS + STATIC_NUM_COLS]).astype(np.float32)

        # Chuẩn hóa z-score đặc trưng chuỗi (chỉ fit trên sub_train)
        X_seq_raw_tr = seq_train[sub_tr_idx]
        X_seq_raw_val = seq_train[val_idx]
        self.seq_mean = X_seq_raw_tr.mean(axis=(0, 1), keepdims=True)
        self.seq_std = np.maximum(X_seq_raw_tr.std(axis=(0, 1), keepdims=True), 1e-5)

        X_seq_tr = ((X_seq_raw_tr - self.seq_mean) / self.seq_std).astype(np.float32)
        X_seq_val = ((X_seq_raw_val - self.seq_mean) / self.seq_std).astype(np.float32)

        M_tr = mask_train[sub_tr_idx].astype(np.float32)
        y_tr = enr_sub['at_risk'].to_numpy(dtype=np.float32)
        y_val = enr_val['at_risk'].to_numpy(dtype=np.float32)

        n_pos = max(float(y_tr.sum()), 1.0)
        n_neg = max(float(len(y_tr) - n_pos), 1.0)
        self.pos_weight_val = n_neg / n_pos

        self.model = EarlyWarningLSTMNet(
            seq_dim=X_seq_tr.shape[-1],
            static_dim=X_stat_tr.shape[-1],
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pos_w_tensor = torch.tensor(self.pos_weight_val, dtype=torch.float32, device=self.device)
        bce_none = nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor, reduction='none')

        # Trọng số thời gian nhấn mạnh thêm các mốc tuần đánh giá chuẩn
        time_weights = np.ones(X_seq_tr.shape[1], dtype=np.float32)
        for w in eval_weeks:
            if 1 <= w <= X_seq_tr.shape[1]:
                time_weights[w - 1] = 1.5
        tw_tensor = torch.tensor(time_weights[None, :], dtype=torch.float32, device=self.device)

        train_ds = TensorDataset(
            torch.from_numpy(X_seq_tr),
            torch.from_numpy(X_stat_tr),
            torch.from_numpy(M_tr),
            torch.from_numpy(y_tr),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        t_seq_val = torch.from_numpy(X_seq_val).to(self.device)
        t_stat_val = torch.from_numpy(X_stat_val).to(self.device)

        best_val_auc = -1.0
        best_state = None
        no_improve = 0

        for _epoch in range(self.max_epochs):
            self.model.train()
            for bx_seq, bx_stat, b_mask, by in train_loader:
                bx_seq = bx_seq.to(self.device)
                bx_stat = bx_stat.to(self.device)
                b_mask = b_mask.to(self.device)
                by_exp = by.to(self.device).unsqueeze(1).expand_as(b_mask)

                optimizer.zero_grad()
                logits = self.model(bx_seq, bx_stat)
                raw_loss = bce_none(logits, by_exp)
                eff_mask = b_mask * tw_tensor
                loss = (raw_loss * eff_mask).sum() / torch.clamp(eff_mask.sum(), min=1.0)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            # Đánh giá trung bình AUC-ROC trên các tuần mốc của tập Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(t_seq_val, t_stat_val).cpu().numpy()
            val_probs = 1.0 / (1.0 + np.exp(-(val_logits - np.log(self.pos_weight_val))))

            aucs = []
            for w in eval_weeks:
                aucs.append(roc_auc_score(y_val, val_probs[:, w - 1]))
            mean_val_auc = float(np.mean(aucs))

            if mean_val_auc > best_val_auc + 1e-4:
                best_val_auc = mean_val_auc
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Tối ưu ngưỡng phân loại F1 trên tập Validation cho từng tuần mốc (mục 4 DSS_plan.docx)
        self.model.eval()
        with torch.no_grad():
            val_logits = self.model(t_seq_val, t_stat_val).cpu().numpy()
        val_probs = 1.0 / (1.0 + np.exp(-(val_logits - np.log(self.pos_weight_val))))

        for w in eval_weeks:
            p_w = val_probs[:, w - 1]
            best_th, best_f1 = 0.42, -1.0
            for th in np.linspace(0.41, 0.56, 31):
                f1 = f1_score(y_val, (p_w >= th).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_th = float(th)
            self.thresholds_by_week[w] = best_th

        return self

    def predict_proba_all_weeks(self, enr_df: pd.DataFrame, seq_arr: np.ndarray) -> np.ndarray:
        """
        Trả về ma trận xác suất đã hiệu chỉnh P(at_risk=1) kích thước (N, max_week).
        """
        X_stat = self.static_prep.transform(enr_df[STATIC_CAT_COLS + STATIC_NUM_COLS]).astype(np.float32)
        X_seq = ((seq_arr - self.seq_mean) / self.seq_std).astype(np.float32)

        self.model.eval()
        probs_list = []
        log_pw = float(np.log(self.pos_weight_val))
        with torch.no_grad():
            for start in range(0, len(X_stat), 1024):
                end = start + 1024
                ts = torch.from_numpy(X_seq[start:end]).to(self.device)
                tb = torch.from_numpy(X_stat[start:end]).to(self.device)
                logits = self.model(ts, tb).cpu().numpy()
                probs = 1.0 / (1.0 + np.exp(-(logits - log_pw)))
                probs_list.append(probs)

        return np.vstack(probs_list)
