"""Interpolate model-predicted PV curves from validation grid to a custom trajectory. volume control"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent  # CardioGraphFENet
MODEL_DIR = BASE_DIR / "output"
TRAINING_SCRIPT = BASE_DIR / "model_train.py"
DATA_ROOT = BASE_DIR / "DATA"
MESH_ID = 0
CUSTOM_TIME_VOLUME = BASE_DIR / "DATA" / "circuit_time_volume.txt"

# Output paths
OUTPUT_DIR = BASE_DIR / "output" / "pv_inference"
OUTPUT_PV_FILE = OUTPUT_DIR / "predicted_pv_loop.txt"
OUTPUT_FIG = OUTPUT_DIR / "predicted_pv_loop.png"
OUTPUT_FIG_3D = OUTPUT_DIR / "predicted_pv_loop_3d.png"
TIME_STRIDE_VIS = 8


def _load_training_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("graphfe_train", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load training module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.modules["train"] = module
    sys.modules["graphfe_v4_7"] = module
    spec.loader.exec_module(module)

    import __main__

    __main__.NormalizationStats = module.NormalizationStats
    __main__.FixedSequenceDataset = module.FixedSequenceDataset
    __main__.MeshStatics = module.MeshStatics
    return module


def _load_dataset_cache(model_dir: Path) -> Dict[str, object]:
    cache_path = model_dir / "dataset_cache.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"{cache_path} not found")
    return torch.load(cache_path, map_location="cpu")


def _collect_mesh_samples(val_dataset, mesh_id: int) -> List[Tuple[int, Dict[str, torch.Tensor]]]:
    samples: List[Tuple[int, Dict[str, torch.Tensor]]] = []
    for idx, (mesh_idx, _) in enumerate(val_dataset.samples):
        if int(mesh_idx) == mesh_id:
            samples.append((idx, val_dataset[idx]))
    if not samples:
        raise RuntimeError(f"No validation samples for mesh{mesh_id:02d}")
    return samples


def _run_prediction_grid(
    train_module,
    mesh_samples: List[Tuple[int, Dict[str, torch.Tensor]]],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    static_encoder_path = MODEL_DIR / "static_encoder" / "static_encoder.pt"
    if static_encoder_path.exists():
        weights = torch.load(static_encoder_path, map_location=device)
        static_encoder = train_module.StaticLatentMLP(feature_dim=3, hidden_dim=128)
        static_encoder.load_state_dict(weights["state_dict"])
        static_encoder.to(device)
        static_encoder.eval()
    else:
        static_encoder = train_module.StaticLatentMLP().to(device)
        static_encoder.eval()

    checkpoint_path = MODEL_DIR / "train" / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = train_module.CGFENet(
        node_in=mesh_samples[0][1]["graph"].x.size(1),
        dynamic_dim=mesh_samples[0][1]["dynamic"].size(1),
        hidden=128,
        apply_sigmoid=train_module.CONFIG.get("normalize_pressure", False),
        fusion_mode=train_module.CONFIG.get("fusion_mode", "layernorm"),
        static_encoder=static_encoder,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    volumes: List[float] = []
    pressure_preds: List[np.ndarray] = []
    time_reference: np.ndarray | None = None

    with torch.no_grad():
        for idx, sample in mesh_samples:
            graph = sample["graph"].clone().to(device)
            dynamic = sample["dynamic"].unsqueeze(0).to(device)
            disp_indices = sample.get("disp_time_indices")
            if disp_indices is not None:
                disp_indices = disp_indices.to(device)
            batch_graph = Batch.from_data_list([graph])
            outputs = model(batch_graph, dynamic, disp_indices=disp_indices)
            pressure_pred = outputs["forward"]["pressure"].squeeze(0).cpu().numpy()

            if train_module.CONFIG.get("normalize_pressure", False):
                pressure_pred *= train_module.CONFIG.get("pressure_scale", 1.0)

            volume_series = sample["volume_values"].cpu().numpy()
            time_series = sample["time_values"].cpu().numpy()

            if time_reference is None:
                time_reference = time_series
            elif not np.allclose(time_reference, time_series, atol=1e-5):
                raise RuntimeError("Validation samples use different time grids; cannot interpolate.")

            volumes.append(float(volume_series[0]))
            pressure_preds.append(pressure_pred)

    volumes_arr = np.asarray(volumes, dtype=np.float64)
    pressure_grid = np.stack(pressure_preds, axis=0)  # [num_volumes, T]
    time_arr = time_reference.astype(np.float64)

    sort_idx = np.argsort(volumes_arr)
    volumes_arr = volumes_arr[sort_idx]
    pressure_grid = pressure_grid[sort_idx]

    return volumes_arr, time_arr, pressure_grid


def _load_custom_time_volume(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    time = data[:, 0]
    volume = data[:, 1]
    return time.astype(np.float64), volume.astype(np.float64)


def _interpolate_prediction(
    volumes_grid: np.ndarray,
    time_grid: np.ndarray,
    pressure_grid: np.ndarray,
    custom_time: np.ndarray,
    custom_volume: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    time_min, time_max = custom_time.min(), custom_time.max()
    val_time_min, val_time_max = time_grid[0], time_grid[-1]
    if time_max <= time_min:
        raise ValueError("Custom time series must span positive duration.")
    normalized_time = (custom_time - time_min) / (time_max - time_min)
    mapped_time = normalized_time * (val_time_max - val_time_min) + val_time_min

    pressures_at_time = np.empty((volumes_grid.shape[0], mapped_time.shape[0]), dtype=np.float64)
    for i, curve in enumerate(pressure_grid):
        pressures_at_time[i] = np.interp(mapped_time, time_grid, curve)

    interp_pressures = np.empty_like(mapped_time)
    for t in range(mapped_time.shape[0]):
        interp_pressures[t] = np.interp(
            custom_volume[t],
            volumes_grid,
            pressures_at_time[:, t],
            left=np.nan,
            right=np.nan,
        )

    return mapped_time, interp_pressures


def _plot_overlay(
    volumes_grid: np.ndarray,
    time_grid: np.ndarray,
    pressure_grid: np.ndarray,
    custom_volume: np.ndarray,
    custom_pressure: np.ndarray,
    custom_time: np.ndarray,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # Set axis limits
    p_min, p_max = custom_pressure.min(), custom_pressure.max()
    p_pad = (p_max - p_min) * 0.1
    ax.set_ylim(p_min - p_pad, p_max + p_pad)
    ax.set_xlim(60, 130)

    sc = ax.scatter(
        custom_volume,
        custom_pressure,
        c=(custom_time - custom_time.min()),
        cmap="turbo",
        s=20,
        edgecolors="none",
    )
    ax.plot(custom_volume, custom_pressure, color="tab:orange", linewidth=1.5, alpha=0.9, label="Interpolated Pred")

    cbar = fig.colorbar(sc, pad=0.02)
    cbar.set_label("Time (ms)")

    ax.set_xlabel("Volume (ml)")
    ax.set_ylabel("Pressure (mmHg)")
    ax.set_title(f"Predicted PV Loop | mesh{MESH_ID:02d}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_overlay_3d(
    volumes_grid: np.ndarray,
    time_grid: np.ndarray,
    pressure_grid: np.ndarray,
    custom_time: np.ndarray,
    custom_volume: np.ndarray,
    custom_pressure: np.ndarray,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    stride = max(1, int(TIME_STRIDE_VIS))
    for vol_curve, press_curve in zip(volumes_grid, pressure_grid):
        ax.plot(time_grid[::stride], np.full_like(time_grid[::stride], vol_curve), press_curve[::stride], color="lightgray", linewidth=0.7, alpha=0.6)

    sc = ax.scatter(
        custom_time,
        custom_volume,
        custom_pressure,
        c=(custom_time - custom_time.min()),
        cmap="turbo",
        s=12,
        edgecolors="none",
    )
    ax.plot(custom_time, custom_volume, custom_pressure, color="tab:orange", linewidth=1.2, alpha=0.9, label="Interpolated Pred")

    cbar = fig.colorbar(sc, pad=0.02)
    cbar.set_label("Custom Time (shifted)")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Volume (ml)")
    ax.set_zlabel("Pressure (mmHg)")
    ax.set_title(f"P-V-T Comparison (Predicted Grid vs Custom Path) | mesh{MESH_ID:02d}")
    ax.view_init(elev=26.0, azim=-56.0)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_module = _load_training_module(TRAINING_SCRIPT)
    train_module.DATA_ROOT = DATA_ROOT

    cache = _load_dataset_cache(MODEL_DIR)
    val_dataset = cache["val_dataset"]
    mesh_samples = _collect_mesh_samples(val_dataset, MESH_ID)

    volumes_grid, time_grid, pressure_grid = _run_prediction_grid(train_module, mesh_samples, device)
    custom_time, custom_volume = _load_custom_time_volume(CUSTOM_TIME_VOLUME)
    mapped_time, interp_pressure = _interpolate_prediction(volumes_grid, time_grid, pressure_grid, custom_time, custom_volume)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        OUTPUT_PV_FILE,
        np.column_stack([custom_time, custom_volume, interp_pressure]),
        header="time\tvolume\tpressure_pred_interp",
        fmt="%.9f\t%.9f\t%.9f",
    )

    _plot_overlay(volumes_grid, time_grid, pressure_grid, custom_volume, interp_pressure, custom_time, OUTPUT_FIG)
    _plot_overlay_3d(volumes_grid, time_grid, pressure_grid, mapped_time, custom_volume, interp_pressure, OUTPUT_FIG_3D)

    summary = {
        "mesh_id": MESH_ID,
        "validation_volume_range": [float(volumes_grid.min()), float(volumes_grid.max())],
        "custom_volume_range": [float(custom_volume.min()), float(custom_volume.max())],
        "output_pv_file": str(OUTPUT_PV_FILE),
        "output_fig": str(OUTPUT_FIG),
        "output_fig_3d": str(OUTPUT_FIG_3D),
        "nan_count": int(np.isnan(interp_pressure).sum()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

