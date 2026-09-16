# -*- coding: utf-8 -*-
"""Minimal reproduction code for M4 DO-CBC Transformer-LSTM."""
import argparse, json, math, os, random, re, warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from fixed_split_io import fixed_indices, prepare_fixed_inputs
warnings.filterwarnings("ignore")

TARGET_FILE="Target.xlsx"; TARGET_SHEET="SH-Clean"
STAGES=["BJ","GJ1","GJ2","RS"]
EPS=1e-9; FEATURE_MODE="spectral_texture"; SEQ_TOP_K=40; STATIC_TOP_K=40; MAX_CORR_FOR_PRUNE=0.98
TARGET_TRANSFORM="log1p"; BATCH_SIZE=32; EPOCHS=450; PATIENCE=60; LEARNING_RATE=8e-4; WEIGHT_DECAY=1e-4; HIDDEN_DIM=64; TRANSFORMER_HEADS=4; TRANSFORMER_LAYERS=1; LSTM_LAYERS=1; DROPOUT=0.20; TARGET_LOSS_WEIGHT=0.50
DEVICE="cuda" if torch.cuda.is_available() else "cpu"; COMPONENTS=["LPC","LB","SPC","SB","GPC","GB"]
CONFIGS={"spectral": {"feature_mode": "spectral_only", "resolution": "3m", "seed": 1813}, "texture": {"feature_mode": "texture_only", "resolution": "1.5m", "seed": 1587}, "combined": {"feature_mode": "spectral_texture", "resolution": "1.5m", "seed": 370}}

def safe_name(x) -> str:
    s = str(x).strip().replace(" ", "_").replace("-", "_").replace("/", "_")
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def strip_stage_suffix(col: str, stage: str) -> str:
    c = safe_name(col)
    c = c.replace(f"_{stage}_WL", "_WL")
    c = re.sub(f"_{stage}$", "", c)
    return c


def is_id_col(col: str) -> bool:
    c = str(col).lower()
    return ("sample" in c) or (c in ["id", "fid", "objectid"])


def pearson_R(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 3:
        return np.nan
    if np.nanstd(y_true[ok]) < 1e-12 or np.nanstd(y_pred[ok]) < 1e-12:
        return np.nan
    return float(np.corrcoef(y_true[ok], y_pred[ok])[0, 1])


def metric_dict(y_true, y_pred, prefix="") -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_pred = sanitize_array(y_pred, fallback=float(np.nanmedian(y_true)))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        prefix + "R": pearson_R(y_true, y_pred),
        prefix + "R2": float(r2_score(y_true, y_pred)),
        prefix + "MAE": float(mean_absolute_error(y_true, y_pred)),
        prefix + "MSE": float(mean_squared_error(y_true, y_pred)),
        prefix + "RMSE": rmse,
        prefix + "RPD": float(np.std(y_true, ddof=1) / rmse) if rmse > 0 else np.nan,
    }


def sanitize_array(x, fallback=0.0, lower=0.0, upper=None):
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=fallback, posinf=fallback, neginf=fallback)
    if lower is not None:
        x = np.maximum(x, lower)
    if upper is not None:
        x = np.minimum(x, upper)
    return x


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def read_target_table(target_path: str) -> pd.DataFrame:
    df = pd.read_excel(target_path, sheet_name=TARGET_SHEET)
    df.columns = [safe_name(c) for c in df.columns]
    if "TSPU" in df.columns and "Target" not in df.columns:
        df = df.rename(columns={"TSPU": "Target"})
    required = ["LPC", "LB", "SPC", "SB", "GPC", "GB"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns {missing} in sheet '{TARGET_SHEET}' of {target_path}")
    if "sample_id" not in df.columns:
        id_candidates = [c for c in df.columns if is_id_col(c)]
        if not id_candidates:
            raise ValueError(f"Sample ID column not found in sheet '{TARGET_SHEET}' of {target_path}")
        df = df.rename(columns={id_candidates[0]: "sample_id"})
    return df

def add_target_formula(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["LPC", "LB", "SPC", "SB", "GPC", "GB"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Target_formula"] = df["LPC"] * df["LB"] + df["SPC"] * df["SB"] + df["GPC"] * df["GB"]
    if "Target" in df.columns:
        df["Target_excel"] = pd.to_numeric(df["Target"], errors="coerce")
        df["Target_diff_formula_minus_excel"] = df["Target_formula"] - df["Target_excel"]
    else:
        df["Target_excel"] = np.nan
        df["Target_diff_formula_minus_excel"] = np.nan
    return df


def find_sheet(xl: pd.ExcelFile, stage: str, kind: str) -> str:
    expected = f"{stage}-{kind}"
    if expected in xl.sheet_names:
        return expected
    raise ValueError(f"Sheet '{expected}' not found. Available sheets: {xl.sheet_names}")

def read_numeric_sheet(path: str, sheet_name: str, stage: str) -> Tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_excel(path, sheet_name=sheet_name)
    id_col = df.columns[0]
    ids = pd.to_numeric(df[id_col], errors="coerce").to_numpy()
    cols = [c for c in df.columns if not is_id_col(c)]
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X.columns = [strip_stage_suffix(c, stage) for c in cols]
    X = X.loc[:, ~X.columns.duplicated()]
    return ids, X


def make_spectral_features(sp: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in sp.columns:
        out[f"SP_RAW_{c}"] = sp[c]
    # Pairwise spectral indices from all bands.
    cols = list(sp.columns)
    for i in range(len(cols) - 1):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            xa, xb = sp[a].astype(float), sp[b].astype(float)
            out[f"SP_ND_{a}_vs_{b}"] = (xa - xb) / (xa + xb + EPS)
            out[f"SP_R_{a}_div_{b}"] = xa / (xb + EPS)
            out[f"SP_D_{a}_minus_{b}"] = xa - xb
    return pd.DataFrame(out).replace([np.inf, -np.inf], np.nan)


def make_texture_features(tx: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in tx.columns:
        x = tx[c].astype(float)
        out[f"TX_asinh_{c}"] = np.arcsinh(x)
        out[f"TX_logabs_{c}"] = np.sign(x) * np.log1p(np.abs(x))
    return pd.DataFrame(out).replace([np.inf, -np.inf], np.nan)


def load_stage_features(fused_path: str, target_ids: np.ndarray) -> Dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(fused_path)
    stage_frames = {}
    for st in STAGES:
        sp_sheet = find_sheet(xl, st, "Spectral")
        ids_sp, sp = read_numeric_sheet(fused_path, sp_sheet, st)
        tx_sheet = find_sheet(xl, st, "Texture")
        ids_tx, tx = read_numeric_sheet(fused_path, tx_sheet, st)

        # align by sample id to target_ids
        sp_order = {int(i): k for k, i in enumerate(ids_sp) if np.isfinite(i)}
        tx_order = {int(i): k for k, i in enumerate(ids_tx) if np.isfinite(i)}
        sp_idx = [sp_order[int(i)] for i in target_ids]
        tx_idx = [tx_order[int(i)] for i in target_ids]
        sp = sp.iloc[sp_idx].reset_index(drop=True)
        tx = tx.iloc[tx_idx].reset_index(drop=True)

        parts = []
        if FEATURE_MODE in ["spectral_only", "spectral_texture"]:
            parts.append(make_spectral_features(sp))
        if FEATURE_MODE in ["texture_only", "spectral_texture"]:
            parts.append(make_texture_features(tx))
        stage_frames[st] = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
    return stage_frames


def build_static_features(stage_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Multi-temporal same-feature indices: late minus early, ratio, normalized difference.
    parts = []
    for late, early in [("RS", "GJ2"), ("GJ2", "GJ1"), ("RS", "BJ")]:
        common = sorted(set(stage_frames[late].columns).intersection(stage_frames[early].columns))
        out = {}
        for c in common:
            xl = stage_frames[late][c].astype(float)
            xe = stage_frames[early][c].astype(float)
            out[f"MT_{late}_minus_{early}__{c}"] = xl - xe
            out[f"MT_{late}_div_{early}__{c}"] = xl / (xe + EPS)
            out[f"MT_ND_{late}_vs_{early}__{c}"] = (xl - xe) / (xl + xe + EPS)
        if out:
            parts.append(pd.DataFrame(out))
    if not parts:
        return pd.DataFrame(index=stage_frames[STAGES[0]].index)
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)


def spearman_scores(X: pd.DataFrame, y: np.ndarray) -> List[Tuple[str, float]]:
    scores = []
    for c in X.columns:
        s = pd.to_numeric(X[c], errors="coerce")
        if s.notna().sum() < max(20, int(0.50 * len(s))):
            continue
        if s.nunique(dropna=True) < 3:
            continue
        sf = s.fillna(s.median())
        try:
            rho = spearmanr(sf, y, nan_policy="omit").correlation
        except Exception:
            rho = np.nan
        if np.isfinite(rho):
            scores.append((c, abs(float(rho))))
    scores.sort(key=lambda z: z[1], reverse=True)
    return scores


def corr_prune(X: pd.DataFrame, candidates: List[str], max_corr: float, max_keep: int) -> List[str]:
    kept = []
    for c in candidates:
        if len(kept) >= max_keep:
            break
        x = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median()).to_numpy()
        ok = True
        for k in kept:
            y = pd.to_numeric(X[k], errors="coerce").fillna(X[k].median()).to_numpy()
            cc = np.corrcoef(x, y)[0, 1]
            if np.isfinite(cc) and abs(cc) >= max_corr:
                ok = False
                break
        if ok:
            kept.append(c)
    return kept


def select_sequence_features(stage_frames: Dict[str, pd.DataFrame], train_idx: np.ndarray, y_train: np.ndarray) -> List[str]:
    # Select feature names that are shared across stages. Score is mean Spearman over stages.
    common = set(stage_frames[STAGES[0]].columns)
    for st in STAGES[1:]:
        common = common.intersection(stage_frames[st].columns)
    common = sorted(common)
    score_map = {}
    for st in STAGES:
        Xst = stage_frames[st].iloc[train_idx][common]
        for c, sc in spearman_scores(Xst, y_train):
            score_map.setdefault(c, []).append(sc)
    ranked = []
    for c in common:
        if c in score_map:
            ranked.append((c, float(np.mean(score_map[c]))))
    ranked.sort(key=lambda z: z[1], reverse=True)
    candidates = [c for c, _ in ranked[:max(SEQ_TOP_K * 8, SEQ_TOP_K + 20)]]
    # Prune using concatenated stages on training set to reduce duplication.
    Xconcat = pd.concat([stage_frames[st].iloc[train_idx][candidates].add_prefix(st + "__") for st in STAGES], axis=1)
    # For pruning among base feature names, use BJ only as a proxy.
    selected = corr_prune(stage_frames["BJ"].iloc[train_idx], candidates, MAX_CORR_FOR_PRUNE, SEQ_TOP_K)
    if len(selected) < SEQ_TOP_K:
        for c in candidates:
            if c not in selected:
                selected.append(c)
            if len(selected) >= SEQ_TOP_K:
                break
    return selected[:SEQ_TOP_K]


def select_static_features(static_df: pd.DataFrame, train_idx: np.ndarray, y_train: np.ndarray) -> List[str]:
    if static_df.shape[1] == 0 or STATIC_TOP_K <= 0:
        return []
    scores = spearman_scores(static_df.iloc[train_idx], y_train)
    candidates = [c for c, _ in scores[:max(STATIC_TOP_K * 8, STATIC_TOP_K + 20)]]
    selected = corr_prune(static_df.iloc[train_idx], candidates, MAX_CORR_FOR_PRUNE, STATIC_TOP_K)
    if len(selected) < STATIC_TOP_K:
        for c in candidates:
            if c not in selected:
                selected.append(c)
            if len(selected) >= STATIC_TOP_K:
                break
    return selected[:STATIC_TOP_K]


def fill_with_train_median(X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    Xtr = X_train.copy()
    Xva = X_val.copy()
    for c in Xtr.columns:
        med = pd.to_numeric(Xtr[c], errors="coerce").median()
        if not np.isfinite(med):
            med = 0.0
        Xtr[c] = pd.to_numeric(Xtr[c], errors="coerce").fillna(med)
        Xva[c] = pd.to_numeric(Xva[c], errors="coerce").fillna(med)
    return Xtr.to_numpy(dtype=np.float32), Xva.to_numpy(dtype=np.float32)


def build_tensors(stage_frames: Dict[str, pd.DataFrame], static_df: pd.DataFrame,
                  train_idx: np.ndarray, val_idx: np.ndarray, seq_features: List[str], static_features: List[str]):
    seq_train_list, seq_val_list = [], []
    for st in STAGES:
        Xtr, Xva = fill_with_train_median(stage_frames[st].iloc[train_idx][seq_features],
                                          stage_frames[st].iloc[val_idx][seq_features])
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr).astype(np.float32)
        Xva = scaler.transform(Xva).astype(np.float32)
        seq_train_list.append(Xtr)
        seq_val_list.append(Xva)
    Xseq_train = np.stack(seq_train_list, axis=1)  # n, t, p
    Xseq_val = np.stack(seq_val_list, axis=1)

    if static_features:
        Xst_tr, Xst_va = fill_with_train_median(static_df.iloc[train_idx][static_features],
                                                static_df.iloc[val_idx][static_features])
        scaler_static = StandardScaler()
        Xst_tr = scaler_static.fit_transform(Xst_tr).astype(np.float32)
        Xst_va = scaler_static.transform(Xst_va).astype(np.float32)
    else:
        Xst_tr = np.zeros((len(train_idx), 0), dtype=np.float32)
        Xst_va = np.zeros((len(val_idx), 0), dtype=np.float32)
    return Xseq_train, Xst_tr, Xseq_val, Xst_va


def y_forward(Y: np.ndarray) -> np.ndarray:
    if TARGET_TRANSFORM == "log1p":
        return np.log1p(np.clip(Y, 0, None))
    if TARGET_TRANSFORM == "sqrt":
        return np.sqrt(np.clip(Y, 0, None))
    return Y.copy()


def y_inverse(Yt: np.ndarray) -> np.ndarray:
    if TARGET_TRANSFORM == "log1p":
        return np.expm1(np.clip(Yt, -20, 20))
    if TARGET_TRANSFORM == "sqrt":
        return np.square(np.maximum(Yt, 0))
    return Yt.copy()


class YScaler:
    def fit(self, Y: np.ndarray):
        Yt = y_forward(Y)
        self.mean_ = np.nanmean(Yt, axis=0)
        self.std_ = np.nanstd(Yt, axis=0)
        self.std_[self.std_ < 1e-12] = 1.0
        return self
    def transform(self, Y: np.ndarray) -> np.ndarray:
        Yt = y_forward(Y)
        return ((Yt - self.mean_) / self.std_).astype(np.float32)
    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        Yt = Z * self.std_ + self.mean_
        return sanitize_array(y_inverse(Yt), fallback=0.0, lower=0.0)


class ComponentTransformerLSTM(nn.Module):
    def __init__(self, seq_dim: int, static_dim: int, hidden_dim=64, n_heads=4, n_layers=1, lstm_layers=1, dropout=0.2):
        super().__init__()
        self.seq_dim = seq_dim
        self.static_dim = static_dim
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(seq_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, len(STAGES), hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        if static_dim > 0:
            self.static_mlp = nn.Sequential(
                nn.Linear(static_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim), nn.GELU()
            )
            final_dim = hidden_dim * 2
        else:
            self.static_mlp = None
            final_dim = hidden_dim
        self.head = nn.Sequential(
            nn.Linear(final_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 6)
        )
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x_seq, x_static):
        h = self.input_proj(x_seq) + self.pos_embed
        h = self.transformer(h)
        out, _ = self.lstm(h)
        h_last = out[:, -1, :]
        if self.static_mlp is not None and x_static.shape[1] > 0:
            hs = self.static_mlp(x_static)
            h_last = torch.cat([h_last, hs], dim=1)
        return self.head(h_last)


def components_to_target(Y: np.ndarray) -> np.ndarray:
    Y = sanitize_array(Y, fallback=0.0, lower=0.0)
    return Y[:, 0] * Y[:, 1] + Y[:, 2] * Y[:, 3] + Y[:, 4] * Y[:, 5]


def train_model(Xseq_train, Xst_train, Y_train, Xseq_val, Xst_val, Y_val, model_seed: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_global_seed(model_seed)
    y_scaler = YScaler().fit(Y_train)
    Ytr_z = y_scaler.transform(Y_train)
    Yva_z = y_scaler.transform(Y_val)
    target_train = components_to_target(Y_train)
    target_std = max(float(np.std(target_train, ddof=1)), 1e-6)

    train_ds = TensorDataset(
        torch.tensor(Xseq_train, dtype=torch.float32),
        torch.tensor(Xst_train, dtype=torch.float32),
        torch.tensor(Ytr_z, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = ComponentTransformerLSTM(
        seq_dim=Xseq_train.shape[2], static_dim=Xst_train.shape[1],
        hidden_dim=HIDDEN_DIM, n_heads=TRANSFORMER_HEADS,
        n_layers=TRANSFORMER_LAYERS, lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()
    best_state = None
    best_val_loss = float("inf")
    wait = 0
    history = []

    Xseq_val_t = torch.tensor(Xseq_val, dtype=torch.float32).to(device)
    Xst_val_t = torch.tensor(Xst_val, dtype=torch.float32).to(device)
    Yva_z_t = torch.tensor(Yva_z, dtype=torch.float32).to(device)
    Yval_orig_t = torch.tensor(Y_val, dtype=torch.float32).to(device)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb_seq, xb_st, yb_z, yb_orig in loader:
            xb_seq, xb_st, yb_z, yb_orig = xb_seq.to(device), xb_st.to(device), yb_z.to(device), yb_orig.to(device)
            optimizer.zero_grad()
            pred_z = model(xb_seq, xb_st)
            loss_comp = mse(pred_z, yb_z)
            # differentiable inverse transform for Target loss
            pred_t = pred_z * torch.tensor(y_scaler.std_, dtype=torch.float32, device=device) + torch.tensor(y_scaler.mean_, dtype=torch.float32, device=device)
            if TARGET_TRANSFORM == "log1p":
                pred_orig = torch.expm1(torch.clamp(pred_t, -20, 20))
            elif TARGET_TRANSFORM == "sqrt":
                pred_orig = torch.square(torch.relu(pred_t))
            else:
                pred_orig = pred_t
            pred_target = pred_orig[:, 0] * pred_orig[:, 1] + pred_orig[:, 2] * pred_orig[:, 3] + pred_orig[:, 4] * pred_orig[:, 5]
            true_target = yb_orig[:, 0] * yb_orig[:, 1] + yb_orig[:, 2] * yb_orig[:, 3] + yb_orig[:, 4] * yb_orig[:, 5]
            loss_target = torch.mean(((pred_target - true_target) / target_std) ** 2)
            loss = loss_comp + TARGET_LOSS_WEIGHT * loss_target
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_z = model(Xseq_val_t, Xst_val_t)
            val_loss_comp = mse(val_z, Yva_z_t)
            val_t = val_z * torch.tensor(y_scaler.std_, dtype=torch.float32, device=device) + torch.tensor(y_scaler.mean_, dtype=torch.float32, device=device)
            if TARGET_TRANSFORM == "log1p":
                val_orig = torch.expm1(torch.clamp(val_t, -20, 20))
            elif TARGET_TRANSFORM == "sqrt":
                val_orig = torch.square(torch.relu(val_t))
            else:
                val_orig = val_t
            val_target_pred = val_orig[:, 0] * val_orig[:, 1] + val_orig[:, 2] * val_orig[:, 3] + val_orig[:, 4] * val_orig[:, 5]
            val_target_true = Yval_orig_t[:, 0] * Yval_orig_t[:, 1] + Yval_orig_t[:, 2] * Yval_orig_t[:, 3] + Yval_orig_t[:, 4] * Yval_orig_t[:, 5]
            val_loss_target = torch.mean(((val_target_pred - val_target_true) / target_std) ** 2)
            val_loss = float((val_loss_comp + TARGET_LOSS_WEIGHT * val_loss_target).detach().cpu())
        train_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, y_scaler, pd.DataFrame(history)


def predict_components(model, y_scaler, Xseq, Xst):
    device = next(model.parameters()).device
    with torch.no_grad():
        z = model(torch.tensor(Xseq, dtype=torch.float32).to(device),
                  torch.tensor(Xst, dtype=torch.float32).to(device)).detach().cpu().numpy()
    return y_scaler.inverse_transform(z)


def mape_percent(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ok = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-12)
    return float(np.mean(np.abs((y_true[ok] - y_pred[ok]) / y_true[ok])) * 100.0) if ok.any() else np.nan


def save_tables(out_dir, sample_ids, train_idx, val_idx, y_train, p_train, y_val, p_val, train_metrics, val_metrics, selected_rows, config):
    os.makedirs(out_dir, exist_ok=True)
    pred = pd.concat([
        pd.DataFrame({"Split":"train", "Sample_ID":sample_ids[train_idx], "Observed":y_train, "Predicted":p_train}),
        pd.DataFrame({"Split":"validation", "Sample_ID":sample_ids[val_idx], "Observed":y_val, "Predicted":p_val}),
    ], ignore_index=True)
    pred["Residual"] = pred["Observed"] - pred["Predicted"]
    pred.to_csv(os.path.join(out_dir, "predictions.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([{"Split":"train", **train_metrics}, {"Split":"validation", **val_metrics}]).to_csv(
        os.path.join(out_dir, "metrics.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(os.path.join(out_dir, "selected_features.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def main(feature):
    global FEATURE_MODE
    cfg=CONFIGS[feature]; FEATURE_MODE=cfg["feature_mode"]
    fused_file=f'{cfg["resolution"]}-Fused.xlsx'; data_dir=prepare_fixed_inputs(__file__,[TARGET_FILE,fused_file])
    target_df=add_target_formula(read_target_table(os.path.join(data_dir,TARGET_FILE)))
    target_df=target_df.dropna(subset=["sample_id"]+COMPONENTS+["Target_formula"]).reset_index(drop=True)
    sample_ids=target_df["sample_id"].astype(int).to_numpy(); Y=target_df[COMPONENTS].astype(float).to_numpy(); y=target_df["Target_formula"].astype(float).to_numpy()
    train_idx,val_idx=fixed_indices(len(target_df)); set_global_seed(int(cfg["seed"]))
    stages=load_stage_features(os.path.join(data_dir,fused_file),sample_ids); static_df=build_static_features(stages)
    seq=select_sequence_features(stages,train_idx,y[train_idx]); sta=select_static_features(static_df,train_idx,y[train_idx])
    Xtr,Sctr,Xva,Scva=build_tensors(stages,static_df,train_idx,val_idx,seq,sta); Ytr,Yva=Y[train_idx],Y[val_idx]
    model,yscaler,history=train_model(Xtr,Sctr,Ytr,Xva,Scva,Yva,int(cfg["seed"]))
    ptr_comp=predict_components(model,yscaler,Xtr,Sctr); pva_comp=predict_components(model,yscaler,Xva,Scva); ptr=components_to_target(ptr_comp); pva=components_to_target(pva_comp)
    tr=metric_dict(y[train_idx],ptr); tr["MAPE"]=mape_percent(y[train_idx],ptr); va=metric_dict(y[val_idx],pva); va["MAPE"]=mape_percent(y[val_idx],pva)
    out_dir=os.path.join("outputs",f"M4_DO-CBC_{feature}")
    selected=[{"Branch":"sequence","Feature":x} for x in seq]+[{"Branch":"static","Feature":x} for x in sta]
    config={"strategy":"DO-CBC","formula":"Target = LPC*LB + SPC*SB + GPC*GB","feature":feature,**cfg,"target_transform":TARGET_TRANSFORM,"seq_topk":SEQ_TOP_K,"static_topk":STATIC_TOP_K,"max_abs_collinearity":MAX_CORR_FOR_PRUNE,
            "model":{"hidden_dim":HIDDEN_DIM,"n_heads":TRANSFORMER_HEADS,"transformer_layers":TRANSFORMER_LAYERS,"lstm_layers":LSTM_LAYERS,"dropout":DROPOUT},
            "training":{"epochs":EPOCHS,"batch_size":BATCH_SIZE,"learning_rate":LEARNING_RATE,"weight_decay":WEIGHT_DECAY,"patience":PATIENCE,"target_loss_weight":TARGET_LOSS_WEIGHT}}
    save_tables(out_dir,sample_ids,train_idx,val_idx,y[train_idx],ptr,y[val_idx],pva,tr,va,selected,config)
    torch.save({"state_dict":model.state_dict(),"config":config,"y_scaler_mean":yscaler.mean_,"y_scaler_std":yscaler.std_,"selected_sequence_features":seq,"selected_static_features":sta},os.path.join(out_dir,"model.pt"))
    print(pd.DataFrame([{"Split":"train",**tr},{"Split":"validation",**va}]).to_string(index=False))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--feature",choices=["spectral","texture","combined"],required=True); main(p.parse_args().feature)
