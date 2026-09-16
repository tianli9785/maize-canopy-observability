# -*- coding: utf-8 -*-
"""Minimal reproduction code for M1 Direct Target Transformer-LSTM."""
import argparse, copy, json, math, os, random, re, warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from fixed_split_io import fixed_indices, prepare_fixed_inputs
warnings.filterwarnings("ignore")

TARGET_FILE="Target.xlsx"; TARGET_SHEET="SH-Clean"; TARGET_VAR="Target"; SAMPLE_ID_COL="Sample_Number"
STAGES=["BJ","GJ1","GJ2","RS"]
FEATURE_MODE="spectral_texture"; SEQ_TOPK_FEATURES=36; STATIC_TOPK_FEATURES=60; MAX_ABS_COLLINEARITY=0.98
TARGET_TRANSFORM="sqrt"; D_MODEL=64; N_HEADS=4; N_TRANSFORMER_LAYERS=2; LSTM_HIDDEN=64; LSTM_LAYERS=1; DROPOUT=0.20; MLP_HIDDEN=96
EPOCHS=450; BATCH_SIZE=24; LEARNING_RATE=8e-4; WEIGHT_DECAY=2e-4; PATIENCE=60; MIN_DELTA=1e-5; GRAD_CLIP=1.0
DEVICE="cuda" if torch.cuda.is_available() else "cpu"; EPS=1e-9
CONFIGS={"spectral": {"feature_mode": "spectral_only", "resolution": "3m", "seed": 136}, "texture": {"feature_mode": "texture_only", "resolution": "3m", "seed": 619}, "combined": {"feature_mode": "spectral_texture", "resolution": "1.5m", "seed": 315}}

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


def pearson_r(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 3:
        return np.nan
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if np.std(y_true) < EPS or np.std(y_pred) < EPS:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def r2_score_np(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 3:
        return np.nan
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - sse / (sst + EPS))


def rpd_np(y_true, y_pred) -> float:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    if rmse <= 0:
        return np.nan
    return float(np.std(y_true, ddof=1) / rmse)


def target_forward(y: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if mode == "raw":
        return y
    if mode == "sqrt":
        return np.sqrt(np.clip(y, 0, None))
    if mode == "log1p":
        return np.log1p(np.clip(y, 0, None))
    if mode == "asinh":
        return np.arcsinh(y)
    raise ValueError(mode)


def target_inverse(z: np.ndarray, mode: str) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -50, 50)  # numerical safety
    if mode == "raw":
        y = z
    elif mode == "sqrt":
        y = np.square(np.maximum(z, 0))
    elif mode == "log1p":
        y = np.expm1(z)
    elif mode == "asinh":
        y = np.sinh(z)
    else:
        raise ValueError(mode)
    return np.clip(y, 0, None)


def metric_dict(y_true, y_pred) -> Dict[str, float]:
    return {
        "R": pearson_r(y_true, y_pred),
        "R2": r2_score_np(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "RPD": rpd_np(y_true, y_pred),
    }


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_target_table(target_path: str) -> pd.DataFrame:
    df = pd.read_excel(target_path, sheet_name=TARGET_SHEET)
    df.columns = [safe_name(c) for c in df.columns]
    if "TSPU" in df.columns and TARGET_VAR not in df.columns:
        df = df.rename(columns={"TSPU": TARGET_VAR})
    if TARGET_VAR not in df.columns:
        raise ValueError(f"Column '{TARGET_VAR}' not found in sheet '{TARGET_SHEET}' of {target_path}")
    if SAMPLE_ID_COL in df.columns:
        df = df.rename(columns={SAMPLE_ID_COL: "sample_id"})
    elif "sample_id" not in df.columns:
        id_candidates = [c for c in df.columns if is_id_col(c)]
        if not id_candidates:
            raise ValueError(f"Sample ID column not found in sheet '{TARGET_SHEET}' of {target_path}")
        df = df.rename(columns={id_candidates[0]: "sample_id"})
    return df

def find_sheet(xl: pd.ExcelFile, stage: str, kind: str) -> str:
    expected = f"{stage}-{kind}"
    if expected in xl.sheet_names:
        return expected
    raise ValueError(f"Sheet '{expected}' not found. Available sheets: {xl.sheet_names}")

def clean_numeric_df(df: pd.DataFrame, stage: str) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    id_col = None
    for c in df.columns:
        if is_id_col(c):
            id_col = c
            break
    ids = None
    if id_col is not None:
        ids = pd.to_numeric(df[id_col], errors="coerce").to_numpy()

    cols = [c for c in df.columns if not is_id_col(c)]
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X.columns = [strip_stage_suffix(c, stage) for c in cols]
    X = X.loc[:, ~X.columns.duplicated()]
    return X, ids


def add_spectral_features(sp: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in sp.columns:
        out[f"RAW_{c}"] = sp[c]
    # All pairwise indices among bands.
    cols = list(sp.columns)
    for i in range(len(cols) - 1):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            xa = sp[a].astype(float)
            xb = sp[b].astype(float)
            out[f"ND_{a}_vs_{b}"] = (xa - xb) / (xa + xb + EPS)
            out[f"R_{a}_div_{b}"] = xa / (xb + EPS)
            out[f"D_{a}_minus_{b}"] = xa - xb
    return pd.DataFrame(out)


def add_texture_features(tx: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in tx.columns:
        x = tx[c].astype(float)
        out[f"TX_asinh_{c}"] = np.arcsinh(x)
        out[f"TX_logabs_{c}"] = np.sign(x) * np.log1p(np.abs(x))
    return pd.DataFrame(out)


def filter_feature_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "spectral_only":
        cols = [c for c in df.columns if not c.startswith("TX_")]
    elif mode == "texture_only":
        cols = [c for c in df.columns if c.startswith("TX_")]
    elif mode == "spectral_texture":
        cols = list(df.columns)
    else:
        raise ValueError(mode)
    return df[cols]


def load_fused_frames(fused_path: str, target_ids: np.ndarray) -> Dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(fused_path)
    stage_frames = {}
    id_reference = None

    for st in STAGES:
        sp_sheet = find_sheet(xl, st, "Spectral")
        tx_sheet = find_sheet(xl, st, "Texture")
        sp_raw = pd.read_excel(fused_path, sheet_name=sp_sheet)
        tx_raw = pd.read_excel(fused_path, sheet_name=tx_sheet)
        sp, ids_sp = clean_numeric_df(sp_raw, st)
        tx, ids_tx = clean_numeric_df(tx_raw, st)

        # Align rows by sample ID when available.
        ids = ids_sp if ids_sp is not None else ids_tx
        if ids is not None:
            order = {int(v): i for i, v in enumerate(ids) if np.isfinite(v)}
            loc = []
            for sid in target_ids:
                if int(sid) not in order:
                    raise ValueError(f"Sample ID {sid} not found in {os.path.basename(fused_path)} sheet {st}.")
                loc.append(order[int(sid)])
            sp = sp.iloc[loc].reset_index(drop=True)
            tx = tx.iloc[loc].reset_index(drop=True)
        else:
            sp = sp.reset_index(drop=True)
            tx = tx.reset_index(drop=True)

        sp_feat = add_spectral_features(sp)
        tx_feat = add_texture_features(tx)
        feat = pd.concat([sp_feat, tx_feat], axis=1).replace([np.inf, -np.inf], np.nan)
        feat = feat.loc[:, ~feat.columns.duplicated()]
        feat = filter_feature_mode(feat, FEATURE_MODE)
        stage_frames[st] = feat

    return stage_frames


def build_static_temporal_features(stage_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Cross-stage indices from common features."""
    parts = []
    pairs = [("RS", "GJ2"), ("GJ2", "GJ1"), ("RS", "BJ")]
    for late, early in pairs:
        common = sorted(set(stage_frames[late].columns).intersection(stage_frames[early].columns))
        out = {}
        for c in common:
            xl = stage_frames[late][c].astype(float)
            xe = stage_frames[early][c].astype(float)
            out[f"{late}_minus_{early}__{c}"] = xl - xe
            out[f"{late}_div_{early}__{c}"] = xl / (xe + EPS)
            out[f"ND_{late}_vs_{early}__{c}"] = (xl - xe) / (xl + xe + EPS)
        if out:
            parts.append(pd.DataFrame(out))
    if parts:
        return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(index=stage_frames[STAGES[0]].index)


def spearman_abs(x: pd.Series, y: np.ndarray) -> float:
    x = pd.to_numeric(x, errors="coerce")
    if x.notna().sum() < max(20, int(0.5 * len(x))):
        return 0.0
    if x.nunique(dropna=True) < 3:
        return 0.0
    vals = x.fillna(x.median()).to_numpy()
    try:
        r = spearmanr(vals, y, nan_policy="omit").correlation
    except Exception:
        return 0.0
    if not np.isfinite(r):
        return 0.0
    return abs(float(r))


def corr_prune_matrix(X: pd.DataFrame, candidates: List[str], max_abs_corr: float, topk: int) -> List[str]:
    kept = []
    for c in candidates:
        if len(kept) >= topk:
            break
        x = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median()).to_numpy(dtype=float)
        ok = True
        for k in kept:
            y = pd.to_numeric(X[k], errors="coerce").fillna(X[k].median()).to_numpy(dtype=float)
            if np.std(x) > EPS and np.std(y) > EPS:
                cc = np.corrcoef(x, y)[0, 1]
                if np.isfinite(cc) and abs(cc) >= max_abs_corr:
                    ok = False
                    break
        if ok:
            kept.append(c)
    return kept


def select_sequence_common_features(stage_frames: Dict[str, pd.DataFrame], train_idx: np.ndarray, y_train_t: np.ndarray) -> List[str]:
    common = set(stage_frames[STAGES[0]].columns)
    for st in STAGES[1:]:
        common = common.intersection(stage_frames[st].columns)
    common = sorted(common)

    scored = []
    for c in common:
        vals = []
        # Score by maximum train correlation across stages.
        for st in STAGES:
            score = spearman_abs(stage_frames[st].iloc[train_idx][c], y_train_t)
            vals.append(score)
        scored.append((c, max(vals), float(np.mean(vals))))
    scored.sort(key=lambda z: (z[1], z[2]), reverse=True)
    candidates = [c for c, _, _ in scored[:max(SEQ_TOPK_FEATURES * 8, 100)]]

    # Prune using concatenated stage values on the training set.
    concat_train = pd.DataFrame()
    for c in candidates:
        arr = []
        for st in STAGES:
            arr.append(stage_frames[st].iloc[train_idx][c].reset_index(drop=True))
        concat_train[c] = pd.concat(arr, axis=0).reset_index(drop=True)
    selected = corr_prune_matrix(concat_train, candidates, MAX_ABS_COLLINEARITY, SEQ_TOPK_FEATURES)

    if len(selected) < SEQ_TOPK_FEATURES:
        for c, _, _ in scored:
            if c not in selected:
                selected.append(c)
            if len(selected) >= SEQ_TOPK_FEATURES:
                break
    return selected[:SEQ_TOPK_FEATURES]


def select_static_features(static_df: pd.DataFrame, train_idx: np.ndarray, y_train_t: np.ndarray) -> List[str]:
    if static_df.shape[1] == 0:
        return []
    scores = [(c, spearman_abs(static_df.iloc[train_idx][c], y_train_t)) for c in static_df.columns]
    scores.sort(key=lambda z: z[1], reverse=True)
    candidates = [c for c, _ in scores[:max(STATIC_TOPK_FEATURES * 8, 120)]]
    selected = corr_prune_matrix(static_df.iloc[train_idx], candidates, MAX_ABS_COLLINEARITY, STATIC_TOPK_FEATURES)
    if len(selected) < STATIC_TOPK_FEATURES:
        for c, _ in scores:
            if c not in selected:
                selected.append(c)
            if len(selected) >= STATIC_TOPK_FEATURES:
                break
    return selected[:STATIC_TOPK_FEATURES]


def fill_median_train_test(X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X_train2 = np.where(np.isfinite(X_train), X_train, med)
    X_val2 = np.where(np.isfinite(X_val), X_val, med)
    return X_train2, X_val2, med


def zscore_train_val(X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(X_train, axis=0)
    sd = np.std(X_train, axis=0)
    sd = np.where((sd < EPS) | (~np.isfinite(sd)), 1.0, sd)
    return (X_train - mu) / sd, (X_val - mu) / sd, mu, sd


def prepare_arrays(stage_frames: Dict[str, pd.DataFrame], static_df: pd.DataFrame,
                   train_idx: np.ndarray, val_idx: np.ndarray, y: np.ndarray) -> Dict:
    y_train_raw = y[train_idx]
    y_val_raw = y[val_idx]
    y_train_t = target_forward(y_train_raw, TARGET_TRANSFORM)
    y_val_t = target_forward(y_val_raw, TARGET_TRANSFORM)

    seq_features = select_sequence_common_features(stage_frames, train_idx, y_train_t)
    static_features = select_static_features(static_df, train_idx, y_train_t)

    def seq_stack(indices):
        arr = []
        for st in STAGES:
            arr.append(stage_frames[st].iloc[indices][seq_features].to_numpy(dtype=float))
        return np.stack(arr, axis=1)  # n × time × features

    Xseq_tr = seq_stack(train_idx)
    Xseq_va = seq_stack(val_idx)

    ntr, nt, nf = Xseq_tr.shape
    Xseq_tr_flat = Xseq_tr.reshape(ntr * nt, nf)
    Xseq_va_flat = Xseq_va.reshape(len(val_idx) * nt, nf)
    Xseq_tr_flat, Xseq_va_flat, seq_med = fill_median_train_test(Xseq_tr_flat, Xseq_va_flat)
    Xseq_tr_flat, Xseq_va_flat, seq_mu, seq_sd = zscore_train_val(Xseq_tr_flat, Xseq_va_flat)
    Xseq_tr = Xseq_tr_flat.reshape(ntr, nt, nf)
    Xseq_va = Xseq_va_flat.reshape(len(val_idx), nt, nf)

    if static_features:
        Xsta_tr = static_df.iloc[train_idx][static_features].to_numpy(dtype=float)
        Xsta_va = static_df.iloc[val_idx][static_features].to_numpy(dtype=float)
        Xsta_tr, Xsta_va, static_med = fill_median_train_test(Xsta_tr, Xsta_va)
        Xsta_tr, Xsta_va, static_mu, static_sd = zscore_train_val(Xsta_tr, Xsta_va)
    else:
        Xsta_tr = np.zeros((len(train_idx), 0), dtype=float)
        Xsta_va = np.zeros((len(val_idx), 0), dtype=float)
        static_med = static_mu = static_sd = np.array([])

    return {
        "Xseq_train": Xseq_tr.astype(np.float32),
        "Xseq_val": Xseq_va.astype(np.float32),
        "Xstatic_train": Xsta_tr.astype(np.float32),
        "Xstatic_val": Xsta_va.astype(np.float32),
        "y_train_raw": y_train_raw.astype(np.float32),
        "y_val_raw": y_val_raw.astype(np.float32),
        "y_train_t": y_train_t.astype(np.float32),
        "y_val_t": y_val_t.astype(np.float32),
        "seq_features": seq_features,
        "static_features": static_features,
        "scaler": {
            "seq_med": seq_med, "seq_mu": seq_mu, "seq_sd": seq_sd,
            "static_med": static_med, "static_mu": static_mu, "static_sd": static_sd,
        },
    }


class TargetDataset(Dataset):
    def __init__(self, Xseq, Xstatic, y):
        self.Xseq = torch.tensor(Xseq, dtype=torch.float32)
        self.Xstatic = torch.tensor(Xstatic, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.Xseq[idx], self.Xstatic[idx], self.y[idx]


class TransformerLSTMRegressor(nn.Module):
    def __init__(self, seq_input_dim: int, static_input_dim: int,
                 d_model: int = 64, n_heads: int = 4,
                 n_transformer_layers: int = 2, lstm_hidden: int = 64,
                 lstm_layers: int = 1, dropout: float = 0.2,
                 mlp_hidden: int = 96, n_stages: int = 4):
        super().__init__()
        self.seq_proj = nn.Linear(seq_input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_stages, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False,
        )
        self.static_input_dim = static_input_dim
        if static_input_dim > 0:
            self.static_mlp = nn.Sequential(
                nn.Linear(static_input_dim, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, mlp_hidden // 2),
                nn.ReLU(),
            )
            fusion_dim = lstm_hidden + mlp_hidden // 2
        else:
            self.static_mlp = None
            fusion_dim = lstm_hidden
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def forward(self, xseq, xstatic):
        x = self.seq_proj(xseq) + self.pos_embedding[:, :xseq.shape[1], :]
        x = self.transformer(x)
        out, _ = self.lstm(x)
        seq_repr = out[:, -1, :]
        if self.static_mlp is not None and xstatic.shape[1] > 0:
            static_repr = self.static_mlp(xstatic)
            z = torch.cat([seq_repr, static_repr], dim=1)
        else:
            z = seq_repr
        return self.head(z)


def train_one_model(data: Dict, seed: int) -> Dict:
    set_all_seeds(seed)
    train_ds = TargetDataset(data["Xseq_train"], data["Xstatic_train"], data["y_train_t"])
    val_ds = TargetDataset(data["Xseq_val"], data["Xstatic_val"], data["y_val_t"])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = TransformerLSTMRegressor(
        seq_input_dim=data["Xseq_train"].shape[2],
        static_input_dim=data["Xstatic_train"].shape[1],
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_transformer_layers=N_TRANSFORMER_LAYERS,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT,
        mlp_hidden=MLP_HIDDEN,
        n_stages=len(STAGES),
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=25, factor=0.6)

    best_state = None
    best_val_loss = float("inf")
    bad = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_losses = []
        for xb_seq, xb_sta, yb in train_loader:
            xb_seq = xb_seq.to(DEVICE)
            xb_sta = xb_sta.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb_seq, xb_sta)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb_seq, xb_sta, yb in val_loader:
                xb_seq = xb_seq.to(DEVICE)
                xb_sta = xb_sta.to(DEVICE)
                yb = yb.to(DEVICE)
                pred = model(xb_seq, xb_sta)
                loss = criterion(pred, yb)
                va_losses.append(float(loss.item()))
        train_loss = float(np.mean(tr_losses))
        val_loss = float(np.mean(va_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        scheduler.step(val_loss)

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    def predict(Xseq, Xsta):
        ds = TargetDataset(Xseq, Xsta, np.zeros(len(Xseq), dtype=np.float32))
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        preds = []
        model.eval()
        with torch.no_grad():
            for xb_seq, xb_sta, _ in loader:
                xb_seq = xb_seq.to(DEVICE)
                xb_sta = xb_sta.to(DEVICE)
                pred = model(xb_seq, xb_sta).cpu().numpy().ravel()
                preds.append(pred)
        return np.concatenate(preds)

    train_pred_t = predict(data["Xseq_train"], data["Xstatic_train"])
    val_pred_t = predict(data["Xseq_val"], data["Xstatic_val"])
    train_pred = target_inverse(train_pred_t, TARGET_TRANSFORM)
    val_pred = target_inverse(val_pred_t, TARGET_TRANSFORM)

    train_metrics = metric_dict(data["y_train_raw"], train_pred)
    val_metrics = metric_dict(data["y_val_raw"], val_pred)

    return {
        "model": model,
        "history": history,
        "train_pred": train_pred,
        "val_pred": val_pred,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }


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
    fused_file=f'{cfg["resolution"]}-Fused.xlsx'
    data_dir=prepare_fixed_inputs(__file__, [TARGET_FILE, fused_file])
    target_df=read_target_table(os.path.join(data_dir,TARGET_FILE)).reset_index(drop=True).dropna(subset=[TARGET_VAR]).reset_index(drop=True)
    y=pd.to_numeric(target_df[TARGET_VAR],errors="coerce").to_numpy(dtype=float)
    sample_ids=pd.to_numeric(target_df.get("sample_id",pd.Series(np.arange(1,len(target_df)+1))),errors="coerce").to_numpy()
    valid=np.isfinite(y); y=y[valid]; sample_ids=sample_ids[valid]
    train_idx,val_idx=fixed_indices(len(y))
    stages=load_fused_frames(os.path.join(data_dir,fused_file),sample_ids)
    static_df=build_static_temporal_features(stages)
    data=prepare_arrays(stages,static_df,train_idx,val_idx,y)
    result=train_one_model(data,int(cfg["seed"]))
    tr={**result["train_metrics"],"MAPE":mape_percent(data["y_train_raw"],result["train_pred"])}
    va={**result["val_metrics"],"MAPE":mape_percent(data["y_val_raw"],result["val_pred"])}
    out_dir=os.path.join("outputs",f"M1_Direct_{feature}")
    selected=[{"Branch":"sequence","Feature":x} for x in data["seq_features"]]+[{"Branch":"static","Feature":x} for x in data["static_features"]]
    config={"strategy":"Direct","feature":feature,**cfg,"target_transform":TARGET_TRANSFORM,"seq_topk":SEQ_TOPK_FEATURES,"static_topk":STATIC_TOPK_FEATURES,"max_abs_collinearity":MAX_ABS_COLLINEARITY,
            "model":{"d_model":D_MODEL,"n_heads":N_HEADS,"transformer_layers":N_TRANSFORMER_LAYERS,"lstm_hidden":LSTM_HIDDEN,"lstm_layers":LSTM_LAYERS,"dropout":DROPOUT,"mlp_hidden":MLP_HIDDEN},
            "training":{"epochs":EPOCHS,"batch_size":BATCH_SIZE,"learning_rate":LEARNING_RATE,"weight_decay":WEIGHT_DECAY,"patience":PATIENCE}}
    save_tables(out_dir,sample_ids,train_idx,val_idx,data["y_train_raw"],result["train_pred"],data["y_val_raw"],result["val_pred"],tr,va,selected,config)
    torch.save({"state_dict":result["model"].state_dict(),"config":config,"selected_sequence_features":data["seq_features"],"selected_static_features":data["static_features"],"scaler":data["scaler"]},os.path.join(out_dir,"model.pt"))
    print(pd.DataFrame([{"Split":"train",**tr},{"Split":"validation",**va}]).to_string(index=False))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--feature",choices=["spectral","texture","combined"],required=True); main(p.parse_args().feature)
