# -*- coding: utf-8 -*-
"""Minimal UAV spectral+texture DO-Sum reproduction code."""
import argparse
import os
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

import M3_DO_Sum as core
from fixed_split_io import fixed_indices, prepare_fixed_inputs

TARGET_FILE = "Target.xlsx"
UAV_FILE = "UAV.xlsx"
STAGES = ["BJ", "GJ", "RS"]
FINAL_SEED = 8
FINAL_MODEL_CFG = {
    "d_model": 96,
    "n_head": 4,
    "n_transformer_layers": 2,
    "lstm_hidden": 96,
    "dropout": 0.25,
    "lr": 0.0006,
    "weight_decay": 0.0001,
}


def _canonical_numeric_frame(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    id_columns = [c for c in df.columns if core.is_id_col(c)]
    if not id_columns:
        raise ValueError("Sample ID column not found in UAV data.")
    id_col = id_columns[0]
    ids = pd.to_numeric(df[id_col], errors="coerce").to_numpy()
    x = df.drop(columns=[id_col]).apply(pd.to_numeric, errors="coerce")
    rename = {}
    for c in x.columns:
        m = re.match(r"(?i)^b(\d+)", str(c).strip())
        rename[c] = f"b{int(m.group(1))}" if m else core.safe_name(c)
    x = x.rename(columns=rename)
    return ids, x.loc[:, ~x.columns.duplicated()]


def _align(frame: pd.DataFrame, ids: np.ndarray, target_ids: np.ndarray) -> pd.DataFrame:
    row_map = {int(v): i for i, v in enumerate(ids) if np.isfinite(v)}
    rows = []
    for sample_id in target_ids:
        key = int(sample_id)
        if key not in row_map:
            raise ValueError(f"Sample ID {key} is missing from UAV.xlsx.")
        rows.append(row_map[key])
    return frame.iloc[rows].reset_index(drop=True)


def load_uav_features(path: str, target_ids: np.ndarray) -> Dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(path)
    stage_frames = {}
    for stage in STAGES:
        spectral_sheet = f"{stage}-Spectral"
        texture_sheet = f"{stage}-Texture"
        if spectral_sheet not in xl.sheet_names or texture_sheet not in xl.sheet_names:
            raise ValueError(
                f"UAV.xlsx must contain both '{spectral_sheet}' and '{texture_sheet}'."
            )
        sp_raw = pd.read_excel(path, sheet_name=spectral_sheet)
        tx_raw = pd.read_excel(path, sheet_name=texture_sheet)
        sp_ids, sp = _canonical_numeric_frame(sp_raw)
        tx_ids, tx = _canonical_numeric_frame(tx_raw)
        sp = _align(sp, sp_ids, target_ids)
        tx = _align(tx, tx_ids, target_ids)
        stage_frames[stage] = pd.concat(
            [core.spectral_features(sp), core.texture_features(tx)], axis=1
        ).replace([np.inf, -np.inf], np.nan)
    return stage_frames


def build_static_features(stage_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    pairs = [("RS", "GJ"), ("GJ", "BJ"), ("RS", "BJ")]
    out = {}
    for late, early in pairs:
        late_df = core.filter_mode_columns(stage_frames[late], "spectral_texture")
        early_df = core.filter_mode_columns(stage_frames[early], "spectral_texture")
        common = sorted(set(late_df.columns).intersection(early_df.columns))
        for feature in common:
            xl = late_df[feature].astype(float)
            xe = early_df[feature].astype(float)
            out[f"MT_{late}_minus_{early}__{feature}"] = xl - xe
            out[f"MT_{late}_div_{early}__{feature}"] = xl / (xe + core.EPS)
            out[f"MT_ND_{late}_vs_{early}__{feature}"] = (
                (xl - xe) / (xl + xe + core.EPS)
            )
    return pd.DataFrame(out).replace([np.inf, -np.inf], np.nan)


def _load_data():
    data_dir = prepare_fixed_inputs(__file__, [TARGET_FILE, UAV_FILE])
    target_df = core.read_target_components(os.path.join(data_dir, TARGET_FILE)).dropna(
        subset=["sample_id"] + core.COMPONENTS + ["Target"]
    ).reset_index(drop=True)
    sample_ids = target_df["sample_id"].astype(int).to_numpy()
    components = target_df[core.COMPONENTS].astype(float).to_numpy()
    target = target_df["Target"].astype(float).to_numpy()
    train_idx, val_idx = fixed_indices(len(target_df))
    stages = load_uav_features(os.path.join(data_dir, UAV_FILE), sample_ids)
    static = build_static_features(stages)
    return sample_ids, components, target, train_idx, val_idx, stages, static


def _frozen_tensors(stages, static, train_idx, val_idx, selected_df):
    seq_train, seq_val = [], []
    for stage in STAGES:
        selected = selected_df.loc[
            (selected_df["Feature_type"] == "sequence") & (selected_df["Stage"] == stage),
            "Feature",
        ].tolist()
        x = core.filter_mode_columns(stages[stage], "spectral_texture")
        xtr = x.iloc[train_idx][selected].copy()
        xva = x.iloc[val_idx][selected].copy()
        med = xtr.median(numeric_only=True)
        xtr = xtr.fillna(med)
        xva = xva.fillna(med)
        scaler = StandardScaler()
        xtr_z = scaler.fit_transform(xtr)
        xva_z = scaler.transform(xva)
        pad_width = core.TOP_SEQ_PER_STAGE - xtr_z.shape[1]
        if pad_width:
            xtr_z = np.pad(xtr_z, ((0, 0), (0, pad_width)), mode="constant")
            xva_z = np.pad(xva_z, ((0, 0), (0, pad_width)), mode="constant")
        seq_train.append(xtr_z)
        seq_val.append(xva_z)

    xseq_train = np.stack(seq_train, axis=1).astype(np.float32)
    xseq_val = np.stack(seq_val, axis=1).astype(np.float32)

    selected_static = selected_df.loc[
        selected_df["Feature_type"] == "static", "Feature"
    ].tolist()
    xtr = static.iloc[train_idx][selected_static].copy()
    xva = static.iloc[val_idx][selected_static].copy()
    med = xtr.median(numeric_only=True)
    xtr = xtr.fillna(med)
    xva = xva.fillna(med)
    scaler = StandardScaler()
    xstatic_train = scaler.fit_transform(xtr).astype(np.float32)
    xstatic_val = scaler.transform(xva).astype(np.float32)
    return xseq_train, xseq_val, xstatic_train, xstatic_val


def reproduce_saved():
    sample_ids, components, target, train_idx, val_idx, stages, static = _load_data()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    artifact_dir = os.path.join(base_dir, "artifacts_M3")
    selected_df = pd.read_csv(os.path.join(artifact_dir, "UAV_features.csv"))
    xseq_train, xseq_val, xstatic_train, xstatic_val = _frozen_tensors(
        stages, static, train_idx, val_idx, selected_df
    )

    checkpoint = torch.load(
        os.path.join(artifact_dir, "UAV_model.pt"),
        map_location=core.DEVICE,
        weights_only=False,
    )
    model_cfg = checkpoint["model_config"]
    model = core.DOsumTransformerLSTMRegressor(
        seq_feature_dim=xseq_train.shape[2],
        static_dim=xstatic_train.shape[1],
        d_model=model_cfg["d_model"],
        n_head=model_cfg["n_head"],
        n_transformer_layers=model_cfg["n_transformer_layers"],
        lstm_hidden=model_cfg["lstm_hidden"],
        lstm_layers=core.LSTM_LAYERS,
        static_hidden=core.STATIC_HIDDEN,
        mlp_hidden=core.MLP_HIDDEN,
        dropout=model_cfg["dropout"],
        n_stages=len(STAGES),
    ).to(core.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pred_train_components = core.predict_components(
        model, xseq_train, xstatic_train, components[train_idx]
    )
    pred_val_components = core.predict_components(
        model, xseq_val, xstatic_val, components[train_idx]
    )
    pred_train = core.components_to_target(pred_train_components)
    pred_val = core.components_to_target(pred_val_components)

    train_metrics = core.calc_metrics(target[train_idx], pred_train)
    val_metrics = core.calc_metrics(target[val_idx], pred_val)
    train_metrics["MAPE"] = core.mape_percent(target[train_idx], pred_train)
    val_metrics["MAPE"] = core.mape_percent(target[val_idx], pred_val)

    out_dir = os.path.join("outputs", "M3_DO-Sum_UAV_combined")
    selected_rows = selected_df.rename(columns={"Feature_type": "Branch"}).to_dict("records")
    config = {
        "strategy": "DO-Sum",
        "observation_source": "UAV",
        "feature": "combined",
        "formula": "Target = LPU + SPU + GPU",
        "seed": FINAL_SEED,
        "model_cfg": model_cfg,
        "reproduction_mode": "frozen_final_checkpoint",
        "best_epoch": checkpoint.get("model_info", {}).get("best_epoch"),
    }
    core.save_tables(
        out_dir,
        sample_ids,
        train_idx,
        val_idx,
        target[train_idx],
        pred_train,
        target[val_idx],
        pred_val,
        train_metrics,
        val_metrics,
        selected_rows,
        config,
    )
    print(pd.DataFrame([
        {"Split": "train", **train_metrics},
        {"Split": "validation", **val_metrics},
    ]).to_string(index=False))


if __name__ == "__main__":
    reproduce_saved()
