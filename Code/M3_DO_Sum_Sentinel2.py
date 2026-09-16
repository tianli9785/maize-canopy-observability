# -*- coding: utf-8 -*-
"""Minimal Sentinel-2 spectral+texture DO-Sum reproduction code."""
import argparse
import json
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
SENTINEL_FILE = "Sentinel.xlsx"
STAGES = ["BJ", "GJ1", "GJ2", "RS"]
STAGE_TAGS = {"BJ": "BJ", "GJ1": "GJ0718", "GJ2": "GJ0827", "RS": "RS"}
BAND_TO_CANONICAL = {
    "Blue": "b1",
    "Green": "b2",
    "Red": "b3",
    "RE1": "b4",
    "RE2": "b5",
    "RE3": "b6",
    "NIR": "b7",
    "RE4": "b8",
    "SWIR1": "b9",
    "SWIR2": "b10",
}
FINAL_SEED = 16
FINAL_MODEL_CFG = {
    "d_model": 96,
    "n_head": 4,
    "n_transformer_layers": 2,
    "lstm_hidden": 96,
    "dropout": 0.25,
    "lr": 0.0006,
    "weight_decay": 0.0001,
}


def _canonical_numeric_frame(df: pd.DataFrame, stage: str, kind: str) -> Tuple[np.ndarray, pd.DataFrame]:
    id_columns = [c for c in df.columns if core.is_id_col(c)]
    if not id_columns:
        raise ValueError(f"Sample ID column not found in {stage}-{kind}.")
    id_col = id_columns[0]
    ids = pd.to_numeric(df[id_col], errors="coerce").to_numpy()
    x = df.drop(columns=[id_col]).apply(pd.to_numeric, errors="coerce")

    if kind == "Spectral":
        rename = {}
        for c in x.columns:
            name = str(c).strip()
            if name in BAND_TO_CANONICAL:
                rename[c] = BAND_TO_CANONICAL[name]
            else:
                clean = core.safe_name(c)
                clean = re.sub(rf"_{re.escape(STAGE_TAGS[stage])}$", "", clean)
                rename[c] = clean
        x = x.rename(columns=rename)
    else:
        tag = STAGE_TAGS[stage]
        rename = {}
        for c in x.columns:
            clean = core.safe_name(c)
            clean = re.sub(rf"_{re.escape(tag)}(?:_W|_WL)?$", "", clean)
            rename[c] = clean.rstrip("_")
        x = x.rename(columns=rename)

    return ids, x.loc[:, ~x.columns.duplicated()]


def _align(frame: pd.DataFrame, ids: np.ndarray, target_ids: np.ndarray) -> pd.DataFrame:
    row_map = {int(v): i for i, v in enumerate(ids) if np.isfinite(v)}
    rows = []
    for sample_id in target_ids:
        key = int(sample_id)
        if key not in row_map:
            raise ValueError(f"Sample ID {key} is missing from Sentinel.xlsx.")
        rows.append(row_map[key])
    return frame.iloc[rows].reset_index(drop=True)


def load_sentinel_features(path: str, target_ids: np.ndarray) -> Dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(path)
    stage_frames = {}
    for stage in STAGES:
        spectral_sheet = f"{stage}-Spectral"
        texture_sheet = f"{stage}-Texture"
        if spectral_sheet not in xl.sheet_names or texture_sheet not in xl.sheet_names:
            raise ValueError(
                f"Sentinel.xlsx must contain both '{spectral_sheet}' and '{texture_sheet}'."
            )

        sp_raw = pd.read_excel(path, sheet_name=spectral_sheet)
        tx_raw = pd.read_excel(path, sheet_name=texture_sheet)
        sp_ids, sp = _canonical_numeric_frame(sp_raw, stage, "Spectral")
        tx_ids, tx = _canonical_numeric_frame(tx_raw, stage, "Texture")
        sp = _align(sp, sp_ids, target_ids)
        tx = _align(tx, tx_ids, target_ids)

        stage_frames[stage] = pd.concat(
            [core.spectral_features(sp), core.texture_features(tx)], axis=1
        ).replace([np.inf, -np.inf], np.nan)
    return stage_frames


def build_static_features(stage_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    pairs = [("RS", "GJ2"), ("GJ2", "GJ1"), ("RS", "BJ")]
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
    data_dir = prepare_fixed_inputs(__file__, [TARGET_FILE, SENTINEL_FILE])
    target_df = core.read_target_components(os.path.join(data_dir, TARGET_FILE)).dropna(
        subset=["sample_id"] + core.COMPONENTS + ["Target"]
    ).reset_index(drop=True)
    sample_ids = target_df["sample_id"].astype(int).to_numpy()
    components = target_df[core.COMPONENTS].astype(float).to_numpy()
    target = target_df["Target"].astype(float).to_numpy()
    train_idx, val_idx = fixed_indices(len(target_df))
    stages = load_sentinel_features(os.path.join(data_dir, SENTINEL_FILE), sample_ids)
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
        seq_train.append(scaler.fit_transform(xtr))
        seq_val.append(scaler.transform(xva))

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
    selected_df = pd.read_csv(os.path.join(artifact_dir, "Sentinel2_features.csv"))
    xseq_train, xseq_val, xstatic_train, xstatic_val = _frozen_tensors(
        stages, static, train_idx, val_idx, selected_df
    )

    checkpoint = torch.load(
        os.path.join(artifact_dir, "Sentinel2_model.pt"),
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

    out_dir = os.path.join("outputs", "M3_DO-Sum_Sentinel2_combined")
    selected_rows = selected_df.rename(columns={"Feature_type": "Branch"}).to_dict("records")
    config = {
        "strategy": "DO-Sum",
        "observation_source": "Sentinel-2",
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


def retrain():
    core.SEQUENCE_FEATURE_MODE = "spectral_texture"
    core.STATIC_FEATURE_MODE = "spectral_texture"
    sample_ids, components, target, train_idx, val_idx, stages, static = _load_data()
    core.set_all_seeds(FINAL_SEED)
    data = core.prepare_tensors_for_split(
        stages, static, components, target, train_idx, val_idx
    )
    model, info, _, _, pred_train, pred_val, _ = core.train_one_model(
        data, FINAL_MODEL_CFG, FINAL_SEED
    )
    train_metrics = core.calc_metrics(target[train_idx], pred_train)
    val_metrics = core.calc_metrics(target[val_idx], pred_val)
    train_metrics["MAPE"] = core.mape_percent(target[train_idx], pred_train)
    val_metrics["MAPE"] = core.mape_percent(target[val_idx], pred_val)

    out_dir = os.path.join("outputs", "M3_DO-Sum_Sentinel2_combined_retrained")
    selected_rows = []
    for stage in STAGES:
        selected_rows.extend(
            {"Branch": "sequence", "Stage": stage, "Feature": f}
            for f in data["stage_selected"][stage]
        )
    selected_rows.extend(
        {"Branch": "static", "Stage": "multi-temporal", "Feature": f}
        for f in data["static_selected"]
    )
    config = {
        "strategy": "DO-Sum",
        "observation_source": "Sentinel-2",
        "feature": "combined",
        "formula": "Target = LPU + SPU + GPU",
        "seed": FINAL_SEED,
        "model_cfg": FINAL_MODEL_CFG,
        "reproduction_mode": "retrained_once",
        "best_epoch": info["best_epoch"],
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
    torch.save(
        {"model_state_dict": model.state_dict(), "model_info": info, "model_config": FINAL_MODEL_CFG},
        os.path.join(out_dir, "model.pt"),
    )
    print(pd.DataFrame([
        {"Split": "train", **train_metrics},
        {"Split": "validation", **val_metrics},
    ]).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain once with the final Sentinel-2 settings instead of loading the frozen final checkpoint.",
    )
    args = parser.parse_args()
    retrain() if args.retrain else reproduce_saved()
