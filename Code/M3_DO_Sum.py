# -*- coding: utf-8 -*-
"""Minimal reproduction code for M3 DO-Sum Transformer-LSTM."""
import argparse, json, math, os, random, re, warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from fixed_split_io import fixed_indices, prepare_fixed_inputs
warnings.filterwarnings("ignore")

TARGET_FILE="Target.xlsx"; TARGET_SHEET="SH-Clean"
STAGES=["BJ","GJ1","GJ2","RS"]
SEQUENCE_FEATURE_MODE="spectral_texture"; STATIC_FEATURE_MODE="spectral_texture"; TOP_SEQ_PER_STAGE=24; TOP_STATIC=36; MAX_ABS_COLLINEARITY=0.98
TARGET_TRANSFORM="sqrt"; EPOCHS=50000; PATIENCE=1000; BATCH_SIZE=32; LEARNING_RATE=8e-4; WEIGHT_DECAY=1e-4; DROPOUT=0.20; D_MODEL=64; N_HEAD=4; N_TRANSFORMER_LAYERS=2; LSTM_HIDDEN=64; LSTM_LAYERS=1; STATIC_HIDDEN=64; MLP_HIDDEN=96; TARGET_LOSS_WEIGHT=0.50; GRAD_CLIP=1.0; EPOCH_LOG_INTERVAL=0
EPS=1e-9; DEVICE="cuda" if torch.cuda.is_available() else "cpu"; COMPONENTS=["LPU","SPU","GPU"]
CONFIGS={"spectral": {"feature_mode": "spectral_only", "resolution": "3m", "seed": 7, "model_cfg": {"d_model": 48, "n_head": 4, "n_transformer_layers": 1, "lstm_hidden": 48, "dropout": 0.15, "lr": 0.001, "weight_decay": 5e-05}}, "texture": {"feature_mode": "texture_only", "resolution": "3m", "seed": 3, "model_cfg": {"d_model": 96, "n_head": 4, "n_transformer_layers": 2, "lstm_hidden": 96, "dropout": 0.25, "lr": 0.0006, "weight_decay": 0.0001}}, "combined": {"feature_mode": "spectral_texture", "resolution": "1.5m", "seed": 5, "model_cfg": {"d_model": 96, "n_head": 4, "n_transformer_layers": 2, "lstm_hidden": 96, "dropout": 0.25, "lr": 0.0006, "weight_decay": 0.0001}}}

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def safe_name(x) -> str:
    s = str(x).strip().replace(" ", "_").replace("-", "_").replace("/", "_")
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def is_id_col(col: str) -> bool:
    c = str(col).lower()
    return ("sample" in c) or (c in ["id", "fid", "objectid"])


def strip_stage_suffix(col: str, stage: str) -> str:
    c = safe_name(col)
    c = c.replace(f"_{stage}_WL", "_WL")
    c = re.sub(f"_{stage}$", "", c)
    return c


def find_sheet(xl: pd.ExcelFile, stage: str, kind: str) -> str:
    expected = f"{stage}-{kind}"
    if expected in xl.sheet_names:
        return expected
    raise ValueError(f"Sheet '{expected}' not found. Available sheets: {xl.sheet_names}")

def pearson_R(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 3:
        return np.nan
    if np.std(y_true[ok]) < EPS or np.std(y_pred[ok]) < EPS:
        return 0.0
    return float(np.corrcoef(y_true[ok], y_pred[ok])[0, 1])


def calc_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) < 3:
        return {"R": np.nan, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "RPD": np.nan}
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "R": pearson_R(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(rmse),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RPD": float(np.std(y_true, ddof=1) / rmse) if rmse > 0 else np.nan,
    }


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
    if mode == "raw":
        return z
    if mode == "sqrt":
        return np.square(np.maximum(z, 0))
    if mode == "log1p":
        z = np.clip(z, -20, 20)
        return np.expm1(z)
    if mode == "asinh":
        z = np.clip(z, -20, 20)
        return np.sinh(z)
    raise ValueError(mode)


def target_inverse_torch(z: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "raw":
        return z
    if mode == "sqrt":
        return torch.square(torch.relu(z))
    if mode == "log1p":
        return torch.expm1(torch.clamp(z, -20, 20))
    if mode == "asinh":
        return torch.sinh(torch.clamp(z, -20, 20))
    raise ValueError(mode)


def sanitize_pred_matrix(pred: np.ndarray, y_train_raw: np.ndarray) -> np.ndarray:
    """Clip component predictions using train-set component distributions only."""
    pred = np.asarray(pred, dtype=float)
    out = pred.copy()
    for j in range(out.shape[1]):
        train_j = y_train_raw[:, j]
        med = float(np.nanmedian(train_j))
        q1, q99 = np.nanpercentile(train_j, [1, 99])
        low = max(0.0, q1 - 1.5 * (q99 - q1))
        high = q99 + 1.5 * (q99 - q1)
        out[:, j][~np.isfinite(out[:, j])] = med
        out[:, j] = np.clip(out[:, j], low, high)
    return out


def components_to_target(Y: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, dtype=float)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.maximum(Y, 0.0)
    return Y[:, 0] + Y[:, 1] + Y[:, 2]


def prefixed_metrics(y_true, y_pred, prefix: str) -> Dict[str, float]:
    m = calc_metrics(y_true, y_pred)
    return {prefix + k: v for k, v in m.items()}


def read_target_components(target_path: str) -> pd.DataFrame:
    df = pd.read_excel(target_path, sheet_name=TARGET_SHEET)
    df.columns = [safe_name(c) for c in df.columns]
    if "TSPU" in df.columns and "Target" not in df.columns:
        df = df.rename(columns={"TSPU": "Target"})
    missing = [c for c in COMPONENTS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns {missing} in sheet '{TARGET_SHEET}' of {target_path}")
    if "sample_id" not in df.columns:
        id_candidates = [c for c in df.columns if is_id_col(c)]
        if not id_candidates:
            raise ValueError(f"Sample ID column not found in sheet '{TARGET_SHEET}' of {target_path}")
        df = df.rename(columns={id_candidates[0]: "sample_id"})
    for c in COMPONENTS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Target"] = df["LPU"] + df["SPU"] + df["GPU"]
    return df.dropna(subset=["sample_id"] + COMPONENTS).reset_index(drop=True)

def clean_numeric_df(df: pd.DataFrame, stage: str) -> Tuple[Optional[np.ndarray], pd.DataFrame]:
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
    return ids, X


def combinations_safe(items, r):
    if r != 2:
        raise ValueError("Only r=2 supported")
    items = list(items)
    for i in range(len(items) - 1):
        for j in range(i + 1, len(items)):
            yield items[i], items[j]


def spectral_features(sp: pd.DataFrame) -> pd.DataFrame:
    out = {}
    cols = list(sp.columns)
    for c in cols:
        out[f"SP_RAW_{c}"] = sp[c].astype(float)
    for a, b in combinations_safe(cols, 2):
        xa = sp[a].astype(float)
        xb = sp[b].astype(float)
        out[f"SP_ND_{a}_vs_{b}"] = (xa - xb) / (xa + xb + EPS)
        out[f"SP_R_{a}_div_{b}"] = xa / (xb + EPS)
        out[f"SP_D_{a}_minus_{b}"] = xa - xb
    return pd.DataFrame(out)


def texture_features(tx: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in tx.columns:
        x = tx[c].astype(float)
        out[f"TX_asinh_{c}"] = np.arcsinh(x)
        out[f"TX_logabs_{c}"] = np.sign(x) * np.log1p(np.abs(x))
    return pd.DataFrame(out)


def filter_mode_columns(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "spectral_only":
        cols = [c for c in df.columns if c.startswith("SP_")]
    elif mode == "texture_only":
        cols = [c for c in df.columns if c.startswith("TX_")]
    elif mode == "spectral_texture":
        cols = [c for c in df.columns if c.startswith("SP_") or c.startswith("TX_")]
    else:
        raise ValueError(mode)
    return df[cols].copy()


def load_fused_resolution(fused_path: str, target_sample_ids: np.ndarray) -> Dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(fused_path)
    stage_frames_all = {}
    for st in STAGES:
        sp_sheet = find_sheet(xl, st, "Spectral")
        tx_sheet = find_sheet(xl, st, "Texture")
        sp_raw = pd.read_excel(fused_path, sheet_name=sp_sheet)
        tx_raw = pd.read_excel(fused_path, sheet_name=tx_sheet)
        sp_ids, sp = clean_numeric_df(sp_raw, st)
        tx_ids, tx = clean_numeric_df(tx_raw, st)
        current_ids = sp_ids if sp_ids is not None else (tx_ids if tx_ids is not None else np.arange(1, len(sp) + 1))
        id_to_row = {int(i): idx for idx, i in enumerate(current_ids) if np.isfinite(i)}
        rows = []
        for sid in target_sample_ids:
            sid_int = int(sid)
            if sid_int not in id_to_row:
                raise ValueError(f"Sample ID {sid_int} is missing from {os.path.basename(fused_path)}")
            rows.append(id_to_row[sid_int])
        sp = sp.iloc[rows].reset_index(drop=True)
        tx = tx.iloc[rows].reset_index(drop=True)
        stage_frames_all[st] = pd.concat([spectral_features(sp), texture_features(tx)], axis=1).replace([np.inf, -np.inf], np.nan)
    return stage_frames_all


def build_static_features(stage_frames: Dict[str, pd.DataFrame], mode: str) -> pd.DataFrame:
    mt_pairs = [("RS", "GJ2"), ("GJ2", "GJ1"), ("RS", "BJ")]
    out = {}
    for late, early in mt_pairs:
        L = filter_mode_columns(stage_frames[late], mode)
        E = filter_mode_columns(stage_frames[early], mode)
        common = sorted(set(L.columns).intersection(E.columns))
        for c in common:
            xl = L[c].astype(float)
            xe = E[c].astype(float)
            out[f"MT_{late}_minus_{early}__{c}"] = xl - xe
            out[f"MT_{late}_div_{early}__{c}"] = xl / (xe + EPS)
            out[f"MT_ND_{late}_vs_{early}__{c}"] = (xl - xe) / (xl + xe + EPS)
    if not out:
        return pd.DataFrame(index=range(len(next(iter(stage_frames.values())))))
    return pd.DataFrame(out).replace([np.inf, -np.inf], np.nan)


def spearman_multi_score(x: pd.Series, Y: np.ndarray) -> float:
    x = pd.to_numeric(x, errors="coerce")
    if x.notna().sum() < max(15, int(0.5 * len(x))):
        return 0.0
    if x.nunique(dropna=True) < 3:
        return 0.0
    xf = x.fillna(x.median()).to_numpy()
    vals = []
    for j in range(Y.shape[1]):
        y = Y[:, j]
        try:
            rho = spearmanr(xf, y, nan_policy="omit").correlation
        except Exception:
            rho = np.nan
        if np.isfinite(rho):
            vals.append(abs(float(rho)))
    return float(np.mean(vals)) if vals else 0.0


def select_by_spearman_multi(Xtr: pd.DataFrame, Ytr_select: np.ndarray, topk: int) -> List[str]:
    scores = []
    for c in Xtr.columns:
        sc = spearman_multi_score(Xtr[c], Ytr_select)
        if sc > 0:
            scores.append((c, sc))
    scores.sort(key=lambda z: z[1], reverse=True)

    selected = []
    for c, _ in scores:
        x = pd.to_numeric(Xtr[c], errors="coerce").fillna(Xtr[c].median()).to_numpy()
        keep = True
        for s in selected:
            y = pd.to_numeric(Xtr[s], errors="coerce").fillna(Xtr[s].median()).to_numpy()
            cc = np.corrcoef(x, y)[0, 1]
            if np.isfinite(cc) and abs(cc) >= MAX_ABS_COLLINEARITY:
                keep = False
                break
        if keep:
            selected.append(c)
        if len(selected) >= topk:
            break
    return selected


def prepare_tensors_for_split(
    stage_frames: Dict[str, pd.DataFrame],
    static_all: pd.DataFrame,
    Y_components: np.ndarray,
    y_target: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Dict:
    # Feature selection uses organ components plus reconstructed Target from training samples only.
    Y_select_train = np.column_stack([Y_components[train_idx], y_target[train_idx]])

    stage_selected = {}
    seq_train_list, seq_val_list = [], []
    for st in STAGES:
        Xst = filter_mode_columns(stage_frames[st], SEQUENCE_FEATURE_MODE)
        selected = select_by_spearman_multi(Xst.iloc[train_idx], Y_select_train, TOP_SEQ_PER_STAGE)
        if len(selected) == 0:
            raise RuntimeError(f"No sequence features were selected for stage {st}")
        stage_selected[st] = selected

        scaler = StandardScaler()
        Xtr = Xst.iloc[train_idx][selected].copy()
        Xva = Xst.iloc[val_idx][selected].copy()
        med = Xtr.median(numeric_only=True)
        Xtr = Xtr.fillna(med)
        Xva = Xva.fillna(med)
        Xtr_z = scaler.fit_transform(Xtr)
        Xva_z = scaler.transform(Xva)
        seq_train_list.append(Xtr_z)
        seq_val_list.append(Xva_z)
        stage_selected[st + "__scaler"] = scaler
        stage_selected[st + "__median"] = med

    Xseq_train = np.stack(seq_train_list, axis=1).astype(np.float32)
    Xseq_val = np.stack(seq_val_list, axis=1).astype(np.float32)

    static_selected = select_by_spearman_multi(static_all.iloc[train_idx], Y_select_train, TOP_STATIC)
    if len(static_selected) == 0:
        Xstatic_train = np.zeros((len(train_idx), 1), dtype=np.float32)
        Xstatic_val = np.zeros((len(val_idx), 1), dtype=np.float32)
        static_scaler = None
        static_median = None
    else:
        static_scaler = StandardScaler()
        Xtr = static_all.iloc[train_idx][static_selected].copy()
        Xva = static_all.iloc[val_idx][static_selected].copy()
        static_median = Xtr.median(numeric_only=True)
        Xtr = Xtr.fillna(static_median)
        Xva = Xva.fillna(static_median)
        Xstatic_train = static_scaler.fit_transform(Xtr).astype(np.float32)
        Xstatic_val = static_scaler.transform(Xva).astype(np.float32)

    Y_train_raw = Y_components[train_idx].astype(float)
    Y_val_raw = Y_components[val_idx].astype(float)
    Y_train_t = target_forward(Y_train_raw, TARGET_TRANSFORM).astype(np.float32)
    Y_val_t = target_forward(Y_val_raw, TARGET_TRANSFORM).astype(np.float32)

    return {
        "Xseq_train": Xseq_train,
        "Xseq_val": Xseq_val,
        "Xstatic_train": Xstatic_train,
        "Xstatic_val": Xstatic_val,
        "Y_train_raw": Y_train_raw,
        "Y_val_raw": Y_val_raw,
        "Y_train_t": Y_train_t,
        "Y_val_t": Y_val_t,
        "y_train_raw": y_target[train_idx].astype(float),
        "y_val_raw": y_target[val_idx].astype(float),
        "stage_selected": stage_selected,
        "static_selected": static_selected,
        "static_scaler": static_scaler,
        "static_median": static_median,
    }


class MultiOutputDataset(Dataset):
    def __init__(self, Xseq, Xstatic, Y_t, Y_raw):
        self.Xseq = torch.tensor(Xseq, dtype=torch.float32)
        self.Xstatic = torch.tensor(Xstatic, dtype=torch.float32)
        self.Y_t = torch.tensor(Y_t, dtype=torch.float32)
        self.Y_raw = torch.tensor(Y_raw, dtype=torch.float32)

    def __len__(self):
        return len(self.Y_t)

    def __getitem__(self, idx):
        return self.Xseq[idx], self.Xstatic[idx], self.Y_t[idx], self.Y_raw[idx]


class DOsumTransformerLSTMRegressor(nn.Module):
    def __init__(
        self,
        seq_feature_dim: int,
        static_dim: int,
        d_model: int = 64,
        n_head: int = 4,
        n_transformer_layers: int = 2,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        static_hidden: int = 64,
        mlp_hidden: int = 96,
        dropout: float = 0.2,
        n_stages: int = 4,
    ):
        super().__init__()
        self.input_proj = nn.Linear(seq_feature_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_stages, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_transformer_layers)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0 if lstm_layers == 1 else dropout,
            bidirectional=False,
        )
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, static_hidden),
            nn.BatchNorm1d(static_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(static_hidden, static_hidden),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + static_hidden, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, len(COMPONENTS)),
        )

    def forward(self, xseq, xstatic):
        h = self.input_proj(xseq) + self.pos_emb
        h = self.transformer(h)
        h, _ = self.lstm(h)
        h_last = h[:, -1, :]
        s = self.static_net(xstatic)
        return self.head(torch.cat([h_last, s], dim=1))


def train_one_model(data: Dict, model_cfg: Dict, seed: int) -> Tuple[nn.Module, Dict, np.ndarray, np.ndarray, List[Dict]]:
    set_all_seeds(seed)

    train_ds = MultiOutputDataset(data["Xseq_train"], data["Xstatic_train"], data["Y_train_t"], data["Y_train_raw"])
    val_ds = MultiOutputDataset(data["Xseq_val"], data["Xstatic_val"], data["Y_val_t"], data["Y_val_raw"])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = DOsumTransformerLSTMRegressor(
        seq_feature_dim=data["Xseq_train"].shape[2],
        static_dim=data["Xstatic_train"].shape[1],
        d_model=model_cfg.get("d_model", D_MODEL),
        n_head=model_cfg.get("n_head", N_HEAD),
        n_transformer_layers=model_cfg.get("n_transformer_layers", N_TRANSFORMER_LAYERS),
        lstm_hidden=model_cfg.get("lstm_hidden", LSTM_HIDDEN),
        lstm_layers=LSTM_LAYERS,
        static_hidden=STATIC_HIDDEN,
        mlp_hidden=MLP_HIDDEN,
        dropout=model_cfg.get("dropout", DROPOUT),
        n_stages=len(STAGES),
    ).to(DEVICE)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=model_cfg.get("lr", LEARNING_RATE),
        weight_decay=model_cfg.get("weight_decay", WEIGHT_DECAY),
    )
    loss_comp_fn = nn.SmoothL1Loss(beta=0.5)
    mse = nn.MSELoss()
    target_train_std = max(float(np.std(data["y_train_raw"], ddof=1)), 1e-6)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_losses = []
        for xb_seq, xb_static, yb_t, yb_raw in train_loader:
            xb_seq = xb_seq.to(DEVICE)
            xb_static = xb_static.to(DEVICE)
            yb_t = yb_t.to(DEVICE)
            yb_raw = yb_raw.to(DEVICE)
            opt.zero_grad()
            pred_t = model(xb_seq, xb_static)
            loss_comp = loss_comp_fn(pred_t, yb_t)
            pred_raw = target_inverse_torch(pred_t, TARGET_TRANSFORM)
            pred_target = torch.sum(pred_raw, dim=1)
            true_target = torch.sum(yb_raw, dim=1)
            loss_target = mse((pred_target - true_target) / target_train_std, torch.zeros_like(true_target))
            loss = loss_comp + TARGET_LOSS_WEIGHT * loss_target
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tr_losses.append(float(loss.detach().cpu()))

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb_seq, xb_static, yb_t, yb_raw in val_loader:
                xb_seq = xb_seq.to(DEVICE)
                xb_static = xb_static.to(DEVICE)
                yb_t = yb_t.to(DEVICE)
                yb_raw = yb_raw.to(DEVICE)
                pred_t = model(xb_seq, xb_static)
                loss_comp = loss_comp_fn(pred_t, yb_t)
                pred_raw = target_inverse_torch(pred_t, TARGET_TRANSFORM)
                pred_target = torch.sum(pred_raw, dim=1)
                true_target = torch.sum(yb_raw, dim=1)
                loss_target = mse((pred_target - true_target) / target_train_std, torch.zeros_like(true_target))
                va_losses.append(float((loss_comp + TARGET_LOSS_WEIGHT * loss_target).detach().cpu()))
        tr_loss = float(np.mean(tr_losses))
        va_loss = float(np.mean(va_losses))
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})

        if EPOCH_LOG_INTERVAL and (epoch == 1 or epoch % EPOCH_LOG_INTERVAL == 0):
            print(
                f"    epoch={epoch:03d}/{EPOCHS} train_loss={tr_loss:.6f} "
                f"val_loss={va_loss:.6f} best_epoch={best_epoch}",
                flush=True,
            )

        if va_loss < best_val_loss - 1e-6:
            best_val_loss = va_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    pred_train_comp = predict_components(model, data["Xseq_train"], data["Xstatic_train"], data["Y_train_raw"])
    pred_val_comp = predict_components(model, data["Xseq_val"], data["Xstatic_val"], data["Y_train_raw"])
    pred_train_target = components_to_target(pred_train_comp)
    pred_val_target = components_to_target(pred_val_comp)

    info = {"best_epoch": best_epoch, "best_val_loss": best_val_loss, **model_cfg}
    return model, info, pred_train_comp, pred_val_comp, pred_train_target, pred_val_target, history


def predict_components(model: nn.Module, Xseq: np.ndarray, Xstatic: np.ndarray, Y_train_raw: np.ndarray) -> np.ndarray:
    ds = MultiOutputDataset(Xseq, Xstatic, np.zeros((len(Xseq), 3), dtype=np.float32), np.zeros((len(Xseq), 3), dtype=np.float32))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb_seq, xb_static, _, _ in loader:
            out = model(xb_seq.to(DEVICE), xb_static.to(DEVICE)).cpu().numpy()
            preds.append(out)
    pred_t = np.vstack(preds)
    pred_raw = target_inverse(pred_t, TARGET_TRANSFORM)
    return sanitize_pred_matrix(pred_raw, Y_train_raw)


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


def train_main(feature):
    global SEQUENCE_FEATURE_MODE, STATIC_FEATURE_MODE
    cfg=CONFIGS[feature]; SEQUENCE_FEATURE_MODE=cfg["feature_mode"]; STATIC_FEATURE_MODE=cfg["feature_mode"]
    fused_file=f'{cfg["resolution"]}-Fused.xlsx'; data_dir=prepare_fixed_inputs(__file__,[TARGET_FILE,fused_file])
    target_df=read_target_components(os.path.join(data_dir,TARGET_FILE)).dropna(subset=["sample_id"]+COMPONENTS+["Target"]).reset_index(drop=True)
    sample_ids=target_df["sample_id"].astype(int).to_numpy(); Y=target_df[COMPONENTS].astype(float).to_numpy(); y=target_df["Target"].astype(float).to_numpy()
    train_idx,val_idx=fixed_indices(len(target_df)); set_all_seeds(int(cfg["seed"]))
    stages=load_fused_resolution(os.path.join(data_dir,fused_file),sample_ids); static_df=build_static_features(stages,STATIC_FEATURE_MODE)
    data=prepare_tensors_for_split(stages,static_df,Y,y,train_idx,val_idx)
    model,info,ptr_comp,pva_comp,ptr,pva,history=train_one_model(data,cfg["model_cfg"],int(cfg["seed"]))
    tr=calc_metrics(y[train_idx],ptr); tr["MAPE"]=mape_percent(y[train_idx],ptr); va=calc_metrics(y[val_idx],pva); va["MAPE"]=mape_percent(y[val_idx],pva)
    out_dir=os.path.join("outputs",f"M3_DO-Sum_{feature}")
    selected=[]
    for st in STAGES:
        selected += [{"Branch":"sequence","Stage":st,"Feature":x} for x in data["stage_selected"][st]]
    selected += [{"Branch":"static","Stage":"multi-temporal","Feature":x} for x in data["static_selected"]]
    config={"strategy":"DO-Sum","formula":"Target = LPU + SPU + GPU","feature":feature,**cfg,"target_transform":TARGET_TRANSFORM,"top_seq_per_stage":TOP_SEQ_PER_STAGE,"top_static":TOP_STATIC,"max_abs_collinearity":MAX_ABS_COLLINEARITY,
            "training":{"epochs":EPOCHS,"batch_size":BATCH_SIZE,"patience":PATIENCE,"target_loss_weight":TARGET_LOSS_WEIGHT},"best_epoch":info["best_epoch"]}
    save_tables(out_dir,sample_ids,train_idx,val_idx,y[train_idx],ptr,y[val_idx],pva,tr,va,selected,config)
    torch.save({"state_dict":model.state_dict(),"config":config,"selected_stage_features":{st:data["stage_selected"][st] for st in STAGES},"selected_static_features":data["static_selected"]},os.path.join(out_dir,"model.pt"))
    print(pd.DataFrame([{"Split":"train",**tr},{"Split":"validation",**va}]).to_string(index=False))



def reproduce_saved(feature):
    """Reproduce the reported M3 DO-Sum result from the frozen final checkpoint."""
    global SEQUENCE_FEATURE_MODE, STATIC_FEATURE_MODE
    cfg = CONFIGS[feature]
    SEQUENCE_FEATURE_MODE = cfg["feature_mode"]
    STATIC_FEATURE_MODE = cfg["feature_mode"]

    fused_file = f'{cfg["resolution"]}-Fused.xlsx'
    data_dir = prepare_fixed_inputs(__file__, [TARGET_FILE, fused_file])
    target_df = read_target_components(os.path.join(data_dir, TARGET_FILE)).dropna(
        subset=["sample_id"] + COMPONENTS + ["Target"]
    ).reset_index(drop=True)
    sample_ids = target_df["sample_id"].astype(int).to_numpy()
    Y = target_df[COMPONENTS].astype(float).to_numpy()
    y = target_df["Target"].astype(float).to_numpy()
    train_idx, val_idx = fixed_indices(len(target_df))

    stages = load_fused_resolution(os.path.join(data_dir, fused_file), sample_ids)
    static_all = build_static_features(stages, STATIC_FEATURE_MODE)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    artifact_dir = os.path.join(base_dir, "artifacts_M3")
    selected_path = os.path.join(artifact_dir, f"{feature}_features.csv")
    model_path = os.path.join(artifact_dir, f"{feature}_model.pt")
    selected_df = pd.read_csv(selected_path)

    seq_train_list, seq_val_list = [], []
    for st in STAGES:
        selected = selected_df.loc[
            (selected_df["Feature_type"] == "sequence") & (selected_df["Stage"] == st),
            "Feature",
        ].tolist()
        Xst = filter_mode_columns(stages[st], SEQUENCE_FEATURE_MODE)
        Xtr = Xst.iloc[train_idx][selected].copy()
        Xva = Xst.iloc[val_idx][selected].copy()
        med = Xtr.median(numeric_only=True)
        Xtr = Xtr.fillna(med)
        Xva = Xva.fillna(med)
        scaler = StandardScaler()
        seq_train_list.append(scaler.fit_transform(Xtr))
        seq_val_list.append(scaler.transform(Xva))

    Xseq_train = np.stack(seq_train_list, axis=1).astype(np.float32)
    Xseq_val = np.stack(seq_val_list, axis=1).astype(np.float32)

    static_selected = selected_df.loc[
        selected_df["Feature_type"] == "static", "Feature"
    ].tolist()
    Xtr = static_all.iloc[train_idx][static_selected].copy()
    Xva = static_all.iloc[val_idx][static_selected].copy()
    med = Xtr.median(numeric_only=True)
    Xtr = Xtr.fillna(med)
    Xva = Xva.fillna(med)
    static_scaler = StandardScaler()
    Xstatic_train = static_scaler.fit_transform(Xtr).astype(np.float32)
    Xstatic_val = static_scaler.transform(Xva).astype(np.float32)

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model_cfg = checkpoint["model_config"]
    model = DOsumTransformerLSTMRegressor(
        seq_feature_dim=Xseq_train.shape[2],
        static_dim=Xstatic_train.shape[1],
        d_model=model_cfg.get("d_model", D_MODEL),
        n_head=model_cfg.get("n_head", N_HEAD),
        n_transformer_layers=model_cfg.get("n_transformer_layers", N_TRANSFORMER_LAYERS),
        lstm_hidden=model_cfg.get("lstm_hidden", LSTM_HIDDEN),
        lstm_layers=LSTM_LAYERS,
        static_hidden=STATIC_HIDDEN,
        mlp_hidden=MLP_HIDDEN,
        dropout=model_cfg.get("dropout", DROPOUT),
        n_stages=len(STAGES),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pred_train_comp = predict_components(model, Xseq_train, Xstatic_train, Y[train_idx])
    pred_val_comp = predict_components(model, Xseq_val, Xstatic_val, Y[train_idx])
    pred_train = components_to_target(pred_train_comp)
    pred_val = components_to_target(pred_val_comp)

    tr = calc_metrics(y[train_idx], pred_train)
    va = calc_metrics(y[val_idx], pred_val)
    tr["MAPE"] = mape_percent(y[train_idx], pred_train)
    va["MAPE"] = mape_percent(y[val_idx], pred_val)

    out_dir = os.path.join("outputs", f"M3_DO-Sum_{feature}")
    selected_rows = selected_df.rename(
        columns={"Feature_type": "Branch"}
    ).to_dict("records")
    config = {
        "strategy": "DO-Sum",
        "formula": "Target = LPU + SPU + GPU",
        "feature": feature,
        **cfg,
        "reproduction_mode": "frozen_final_checkpoint",
        "best_epoch": checkpoint.get("model_info", {}).get("best_epoch"),
    }
    save_tables(
        out_dir, sample_ids, train_idx, val_idx,
        y[train_idx], pred_train, y[val_idx], pred_val,
        tr, va, selected_rows, config,
    )
    print(pd.DataFrame([
        {"Split": "train", **tr},
        {"Split": "validation", **va},
    ]).to_string(index=False))


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feature", choices=["spectral", "texture", "combined"], required=True)
    p.add_argument("--retrain", action="store_true", help="Retrain from scratch instead of reproducing the frozen reported checkpoint.")
    args = p.parse_args()
    if args.retrain:
        train_main(args.feature)
    else:
        reproduce_saved(args.feature)
