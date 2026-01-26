"""CGFENet: shared encoder with dual decoders for pressure and displacement.

The model combines the GraphFusion encoder
with the dual-head displacement/pressure design inspired by the GATv2 cycle
model. Displacement outputs are regularised via a surface-based volume
constraint and provide diagnostic plotting utilities for epi/endo segmentation
and predicted LV cavity volumes.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Configure temporary directory relative to current folder to avoid /tmp space issues
TMP_DIR = str(BASE_DIR / 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)
os.environ['TMPDIR'] = TMP_DIR
os.environ['TEMP'] = TMP_DIR
os.environ['TMP'] = TMP_DIR
tempfile.tempdir = TMP_DIR

import argparse
import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from collections import defaultdict
import re
import sys
import xml.etree.ElementTree as ET

import h5py
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool
from torch_geometric.utils import to_undirected
from tqdm import tqdm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------
DATA_ROOT = BASE_DIR / "DATA"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Separate output directories for each stage
STATIC_OUTPUT_DIR = OUTPUT_DIR / "static_encoder"
TRAIN_OUTPUT_DIR = OUTPUT_DIR / "train"
TEST_OUTPUT_DIR = OUTPUT_DIR / "test"
STATIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VTU_NAME = "u_disp000000.vtu"
H5_NAME = "disp_fields.h5"
METRICS_NAME = "lv_wall_metrics.txt"
FIXED_SEQUENCE_LENGTH = 800
BASE_TOL = 1e-4
DISP_OVERLAP_THRESHOLD_CM = 0.1

# Encoder hyper-parameters (conservative defaults to avoid NaNs)
HIDDEN_CHANNELS = 128
DROPOUT_RATE = 0.1
GLOBAL_FUSION_HEADS = 4
GAT_HEADS = 4

CONFIG = {
    "batch_size": 8,
    "time_stride": 1,
    "num_epochs": 1000,
    "learning_rate": 1e-3,
    "eta_min": 1e-7,
    "weight_decay": 1e-3,
    "max_grad_norm": 2.0,
    "normalize_pressure": False,
    "pressure_scale": 300.0,
    "fusion_mode": "layernorm",  # options: layernorm, linear
    "time_period": 800.0,
    "train_ratio": 0.75,
    "split_seed": 42,
    "random_seed": 42,
    "pressure_loss_weight": 1.0,
    "disp_loss_weight": 1e4,
    "latent_loss_weight": 0.1,
    "plot_every": 10,
    "checkpoint_every": 25,
    "test_mode": False,
    "disp_time_gap": 16,
    "detailed_every": 20,
    "overlay_cases_limit": 5,
    # test-time toggles
    "test_collect_summary": False,  # set True for full per-graph metrics
    "test_overlay_cases_limit": 10,  # override overlay cases in test
    "test_batch_size": 2,           # smaller batch to avoid OOM in test
    "test_eval_disable_inverse": False,  # enable inverse branch during evaluate in test
    # overlay view controls
    "overlay_show_loaded": True,
    "overlay_show_unloaded": False,
    "test_export_stl": True,   # export STL meshes for GT/Pred (Loaded/Unloaded)
    "test_export_vtk": False,  # export legacy VTK (disable by default)
    "overlay_time_stride": 100,
    "overlay_max_frames": 4,
    # plotting sampling controls
    "summary_cases_per_mesh": 1,   # how many target volumes per mesh in summary plots
    "plot_time_stride": 8,         # stride to subsample time series when plotting
    # summary speed control
    "test_summary_graph_limit": 20,  # limit graphs for expensive Hausdorff calculations
    "hd_chunk": 1024,               # cdist chunk size for Hausdorff to limit memory
    "collect_val_summary": False,   # compute per-graph summary during validation when enabled
    "skip_validation": True,        # skip validation during training
    # static encoder options
    "freeze_static_head": True,
    "static_encoder_type": "mlp",   # choices: mlp, graph
    "static_encoder_path": None,
    "pretrain_static_epochs": 1000,
    "pretrain_static_batch_size": 4,
    "pretrain_static_learning_rate": 1e-3,
    "pretrain_static_weight_decay": 0.0,
    "pretrain_static_patience": 1000,
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BEST_MODEL_METRIC = "train_loss"

# Training outputs
MODEL_PATH = TRAIN_OUTPUT_DIR / "best_model.pt"
LOGGER_PATH = TRAIN_OUTPUT_DIR / "train.log"
CHECKPOINT_PATH = TRAIN_OUTPUT_DIR / "checkpoint.pt"
DATASET_CACHE_PATH = OUTPUT_DIR / "dataset_cache.pt"  # shared cache

# Test outputs
TEST_LOGGER_PATH = TEST_OUTPUT_DIR / "test.log"

# Static encoder outputs
STATIC_PRETRAIN_DIR = STATIC_OUTPUT_DIR / "pretrain_logs"
DEFAULT_STATIC_ENCODER_PATH = STATIC_OUTPUT_DIR / "static_encoder.pt"
STATIC_LOGGER_PATH = STATIC_OUTPUT_DIR / "static_pretrain.log"

SUMMARY_SECTIONS: List[Tuple[str, str, Optional[str]]] = [
    ("Forward pressure RMSE (mmHg)", "forward_pressure_rmse", None),
    ("Forward pressure r^2", "forward_pressure_r", "square"),
    ("Forward displacement RMSE (cm)", "forward_disp_rmse", None),
    ("Forward overlap@0.1cm", "forward_disp_overlap_0p1", None),
    ("Forward overlap@0.2cm", "forward_disp_overlap_0p2", None),
    ("Forward HD (cm)", "forward_disp_hd", None),
    ("Inverse displacement RMSE (cm)", "inverse_disp_rmse", None),
    ("Inverse overlap@0.1cm", "inverse_disp_overlap_0p1", None),
    ("Inverse overlap@0.2cm", "inverse_disp_overlap_0p2", None),
    ("Inverse HD (cm)", "inverse_disp_hd", None),
]


# Ensure cached pickles referencing this module can be loaded when executed as script
sys.modules.setdefault("cgfenet", sys.modules[__name__])

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def setup_logger(log_path: Path = LOGGER_PATH, name: str = "CGFENet") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.propagate = False
    logger.info("\n" + "=" * 120)
    logger.info(f"{name} run @ {datetime.now().isoformat()} | device={DEVICE}")
    return logger


def setup_static_pretrain_logger() -> logging.Logger:
    STATIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("StaticEncoder")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(STATIC_LOGGER_PATH, mode="a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.propagate = False
    logger.info("\n" + "=" * 120)
    logger.info(f"Static encoder pretraining run @ {datetime.now().isoformat()} | device={DEVICE}")
    return logger


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For CUDA >= 10.2, ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Mesh utilities
# ---------------------------------------------------------------------------


@dataclass
class MeshStatics:
    coords: np.ndarray
    edge_index: torch.Tensor
    endo_mask: np.ndarray
    epi_mask: np.ndarray
    endo_faces: np.ndarray
    epi_faces: np.ndarray
    static_scalars: np.ndarray


def load_tetra_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    tree = ET.parse(str(path))
    piece = tree.getroot().find("UnstructuredGrid").find("Piece")

    coords_text = (piece.find("Points").find("DataArray").text or "").strip()
    coords = np.fromstring(coords_text, sep=" ", dtype=np.float64).reshape(-1, 3)

    connectivity = None
    offsets = None
    types = None
    for da in piece.find("Cells").findall("DataArray"):
        name = da.attrib.get("Name")
        if name == "connectivity":
            connectivity = np.fromstring((da.text or "").strip(), sep=" ", dtype=np.int64)
        elif name == "offsets":
            offsets = np.fromstring((da.text or "").strip(), sep=" ", dtype=np.int64)
        elif name == "types":
            types = np.fromstring((da.text or "").strip(), sep=" ", dtype=np.uint8)
    if connectivity is None or offsets is None or types is None:
        raise RuntimeError(f"VTU file {path} missing cell connectivity data")

    tets: List[Tuple[int, int, int, int]] = []
    start = 0
    for cell_type, end in zip(types, offsets):
        if cell_type != 10:
            raise ValueError(f"Encountered non-tetra cell type {cell_type} in {path}")
        tet = tuple(connectivity[start:end])
        if len(tet) != 4:
            raise ValueError(f"Expected tetrahedron, got {len(tet)} nodes in {path}")
        tets.append(tet)
        start = end
    return coords, np.asarray(tets, dtype=np.int64)


def extract_boundary_faces(tets: np.ndarray) -> np.ndarray:
    face_map: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], int]] = {}
    for tet in tets:
        a, b, c, d = map(int, tet)
        faces = [
            ((a, b, c), d),
            ((a, c, d), b),
            ((a, d, b), c),
            ((b, d, c), a),
        ]
        for face, opp in faces:
            key = tuple(sorted(face))
            if key in face_map:
                face_map.pop(key)
            else:
                face_map[key] = (face, opp)

    oriented: List[Tuple[int, int, int]] = []
    for face, opp in face_map.values():
        i, j, k = face
        oriented.append((i, j, k))
    return np.asarray(oriented, dtype=np.int64)


def component_area(faces: np.ndarray, vertices: np.ndarray) -> float:
    if faces.size == 0:
        return 0.0
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    cross_prod = np.cross(v2 - v1, v3 - v1)
    return float(0.5 * np.linalg.norm(cross_prod, axis=1).sum())


def split_components(faces: np.ndarray) -> List[np.ndarray]:
    if faces.size == 0:
        return []
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    for idx, (i, j, k) in enumerate(faces):
        for a, b in ((i, j), (j, k), (k, i)):
            key = tuple(sorted((int(a), int(b))))
            edge_to_faces.setdefault(key, []).append(idx)

    adjacency: List[List[int]] = [[] for _ in range(len(faces))]
    for face_indices in edge_to_faces.values():
        if len(face_indices) == 2:
            a, b = face_indices
            adjacency[a].append(b)
            adjacency[b].append(a)

    visited = [False] * len(faces)
    components: List[np.ndarray] = []
    for idx in range(len(faces)):
        if visited[idx]:
            continue
        stack = [idx]
        visited[idx] = True
        comp_ids: List[int] = []
        while stack:
            current = stack.pop()
            comp_ids.append(current)
            for nb in adjacency[current]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        components.append(faces[np.asarray(comp_ids, dtype=np.int64)])
    return components


def select_endocardial_component(components: List[np.ndarray], vertices: np.ndarray) -> np.ndarray:
    if not components:
        raise RuntimeError("No surface components available to select endocardium")
    areas = [component_area(comp, vertices) for comp in components]
    idx = int(np.argmin(areas))
    return components[idx]


def select_epicardial_component(components: List[np.ndarray], vertices: np.ndarray) -> np.ndarray:
    if len(components) < 2:
        raise RuntimeError("Epicardial component not found; need at least two components")
    areas = [component_area(comp, vertices) for comp in components]
    idx = int(np.argmax(areas))
    return components[idx]


def load_wall_metrics(mesh_dir: Path) -> np.ndarray:
    metrics_path = mesh_dir / METRICS_NAME
    scalars = np.zeros(4, dtype=np.float64)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if not line:
                continue
            if line.startswith("lv volume"):
                scalars[0] = float(line.split(":", 1)[1].split()[0])
            elif line.startswith("wall thickness mean"):
                scalars[1] = float(line.split(":", 1)[1].split()[0])
            elif line.startswith("wall thickness min"):
                scalars[2] = float(line.split(":", 1)[1].split()[0])
            elif line.startswith("wall thickness max"):
                scalars[3] = float(line.split(":", 1)[1].split()[0])
    return scalars


def build_edge_index(num_nodes: int, tets: np.ndarray) -> torch.Tensor:
    edge_pairs: List[Tuple[int, int]] = []
    for tet in tets:
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = int(tet[i]), int(tet[j])
                edge_pairs.append((a, b))
                edge_pairs.append((b, a))
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return edge_index


def extract_static_mesh_info(mesh_dir: Path) -> MeshStatics:
    coords, tets = load_tetra_mesh(mesh_dir / VTU_NAME)
    faces = extract_boundary_faces(tets)

    z = coords[:, 2]
    z_max = float(z.max())
    base_mask = np.all(np.abs(z[faces] - z_max) <= BASE_TOL, axis=1)
    trimmed_faces = faces[~base_mask]
    components = split_components(trimmed_faces)
    if not components:
        raise RuntimeError(f"{mesh_dir.name}: empty boundary after base trimming")

    endo_faces = select_endocardial_component(components, coords)
    epi_faces = select_epicardial_component(components, coords)

    endo_vertices = np.unique(endo_faces)
    epi_vertices = np.unique(epi_faces)

    endo_mask = np.zeros(coords.shape[0], dtype=np.float32)
    epi_mask = np.zeros(coords.shape[0], dtype=np.float32)
    endo_mask[endo_vertices] = 1.0
    epi_mask[epi_vertices] = 1.0

    statics = MeshStatics(
        coords=coords.astype(np.float32),
        edge_index=build_edge_index(coords.shape[0], tets),
        endo_mask=endo_mask,
        epi_mask=epi_mask,
        endo_faces=endo_faces.astype(np.int64),
        epi_faces=epi_faces.astype(np.int64),
        static_scalars=np.zeros(4, dtype=np.float64),
    )
    return statics


def safe_extract_static_mesh_info(mesh_dir: Path) -> Optional[MeshStatics]:
    try:
        statics = extract_static_mesh_info(mesh_dir)
        statics.static_scalars = load_wall_metrics(mesh_dir)
        return statics
    except Exception as exc:
        logging.getLogger("CGFENet").warning(f"Skip {mesh_dir.name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class NormalizationStats:
    coord_mean: np.ndarray
    coord_std: np.ndarray
    volume_mean: float
    volume_std: float
    pressure_scale: float
    static_mean: np.ndarray
    static_std: np.ndarray


class FixedSequenceDataset(Dataset):
    """Dataset providing 800-frame sequences conditioned on target volume."""

    def __init__(
        self,
        mesh_indices: Sequence[int],
        stats: NormalizationStats,
        time_stride: int,
        normalize_pressure: bool,
        pressure_scale: float,
        disp_time_gap: int = 1,
        store_disp_full: bool = True,
    ) -> None:
        self.mesh_indices = list(mesh_indices)
        self.stats = stats
        self.time_stride = max(1, time_stride)
        self.normalize_pressure = bool(normalize_pressure)
        self.pressure_scale = float(pressure_scale) if pressure_scale > 0 else 1.0
        self.disp_time_gap = max(1, int(disp_time_gap))
        self.store_disp_full = bool(store_disp_full)

        self.mesh_cache: Dict[int, MeshStatics] = {}
        self.mesh_h5_paths: Dict[int, Path] = {}
        self.samples: List[Tuple[int, float]] = []
        self.preprocessed: List[Dict[str, torch.Tensor]] = []

        for mesh_idx in self.mesh_indices:
            mesh_dir = DATA_ROOT / f"mesh{mesh_idx:02d}"
            statics = safe_extract_static_mesh_info(mesh_dir)
            if statics is None:
                continue
            self.mesh_cache[mesh_idx] = statics
            self.mesh_h5_paths[mesh_idx] = mesh_dir / H5_NAME

            with h5py.File(self.mesh_h5_paths[mesh_idx], "r") as f:
                volume_array = np.asarray(f["volume"], dtype=np.float64)
                unique_volumes = np.unique(volume_array)
            for i in range(0, len(unique_volumes), self.time_stride):
                self.samples.append((mesh_idx, float(unique_volumes[i])))

        for mesh_idx, target_volume in self.samples:
            self.preprocessed.append(self._build_sample(mesh_idx, target_volume))

    def __len__(self) -> int:
        return len(self.preprocessed)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.preprocessed[idx]

    def _build_sample(self, mesh_idx: int, target_volume: float) -> Dict[str, torch.Tensor]:
        statics = self.mesh_cache[mesh_idx]
        coords = statics.coords

        with h5py.File(self.mesh_h5_paths[mesh_idx], "r") as f:
            volume_full = np.asarray(f["volume"], dtype=np.float64)
            mask = volume_full == target_volume
            if not np.any(mask):
                raise RuntimeError(f"Volume {target_volume} not found for mesh {mesh_idx}")

            t_seq = np.asarray(f["t"], dtype=np.float64)[mask]
            volume_seq = volume_full[mask]
            pressure_seq = np.asarray(f["pressure"], dtype=np.float64)[mask]
            disp_seq = np.asarray(f["U"], dtype=np.float32)[mask]

        actual_length = len(t_seq)
        if actual_length == 0:
            raise RuntimeError(f"Empty sequence for mesh {mesh_idx} volume {target_volume}")

        endo_indices = np.where(statics.endo_mask > 0.5)[0]
        if actual_length < FIXED_SEQUENCE_LENGTH:
            pad_len = FIXED_SEQUENCE_LENGTH - actual_length
            if actual_length > 1:
                dt = t_seq[-1] - t_seq[-2]
                t_pad = t_seq[-1] + dt * np.arange(1, pad_len + 1, dtype=np.float64)
            else:
                t_pad = t_seq[-1] + np.arange(1, pad_len + 1, dtype=np.float64)
            volume_pad = np.full(pad_len, volume_seq[-1], dtype=np.float64)
            pressure_pad = np.full(pad_len, pressure_seq[-1], dtype=np.float64)
            disp_pad = np.repeat(disp_seq[-1:, :, :], pad_len, axis=0)

            t_seq = np.concatenate([t_seq, t_pad])
            volume_seq = np.concatenate([volume_seq, volume_pad])
            pressure_seq = np.concatenate([pressure_seq, pressure_pad])
            disp_seq = np.concatenate([disp_seq, disp_pad], axis=0)
        elif actual_length > FIXED_SEQUENCE_LENGTH:
            t_seq = t_seq[:FIXED_SEQUENCE_LENGTH]
            volume_seq = volume_seq[:FIXED_SEQUENCE_LENGTH]
            pressure_seq = pressure_seq[:FIXED_SEQUENCE_LENGTH]
            disp_seq = disp_seq[:FIXED_SEQUENCE_LENGTH]

        stats = self.stats
        coord_mean = stats.coord_mean.astype(np.float32)
        coord_std = np.where(stats.coord_std.astype(np.float32) < 1e-6, 1.0, stats.coord_std.astype(np.float32))
        coords_norm = (coords - coord_mean) / coord_std

        static_mean = stats.static_mean.astype(np.float32)
        static_std = np.where(stats.static_std.astype(np.float32) < 1e-6, 1.0, stats.static_std.astype(np.float32))
        static_norm = (statics.static_scalars.astype(np.float32) - static_mean) / static_std
        static_scalars = np.tile(static_norm, (coords.shape[0], 1))

        phase = (t_seq % CONFIG["time_period"]) / CONFIG["time_period"]
        time_sin = np.sin(2.0 * np.pi * phase)
        time_cos = np.cos(2.0 * np.pi * phase)
        unloaded_volume = float(statics.static_scalars[0])
        delta_volume = volume_seq - unloaded_volume
        volume_mean = stats.volume_mean
        volume_std = stats.volume_std if stats.volume_std >= 1e-6 else 1.0
        volume_norm = (volume_seq - volume_mean) / volume_std
        delta_volume_norm = delta_volume / volume_std
        dynamic_inputs = np.stack([volume_norm, delta_volume_norm, time_sin, time_cos], axis=1).astype(np.float32)

        region_flag = np.full((coords.shape[0], 1), 2.0, dtype=np.float32)
        region_flag[statics.epi_mask > 0.5] = 1.0
        region_flag[statics.endo_mask > 0.5] = 0.0
        node_features = np.concatenate(
            [
                coords_norm,
                region_flag,
                static_scalars,
            ],
            axis=1,
        ).astype(np.float32)

        pressure_values = np.maximum(pressure_seq, 0.0)
        if self.normalize_pressure:
            pressure_values = pressure_values / self.pressure_scale

        coord_std = np.where(self.stats.coord_std < 1e-6, 1.0, self.stats.coord_std).astype(np.float32)
        disp_norm_full = (disp_seq / coord_std).astype(np.float32)  # [T, N, 3]
        disp_norm_full = np.transpose(disp_norm_full, (1, 0, 2))  # [N, T, 3]
        disp_denorm_full = np.transpose(disp_seq.astype(np.float32), (1, 0, 2))

        disp_indices = np.arange(0, FIXED_SEQUENCE_LENGTH, self.disp_time_gap, dtype=np.int64)
        if self.store_disp_full:
            disp_target = torch.from_numpy(disp_norm_full)
            disp_target_cm = torch.from_numpy(disp_denorm_full)
        else:
            disp_target = torch.from_numpy(disp_norm_full[:, disp_indices, :])
            disp_target_cm = None

        return {
            "graph": Data(
                x=torch.from_numpy(node_features),
                edge_index=statics.edge_index,
                pos=torch.from_numpy(coords),
                endo_mask=torch.from_numpy(statics.endo_mask.astype(np.float32)),
            ),
            "dynamic": torch.from_numpy(dynamic_inputs),
            "pressure_target": torch.from_numpy(pressure_values.astype(np.float32)),
            "mesh_idx": mesh_idx,
            "time_values": torch.from_numpy(t_seq.astype(np.float32)),
            "volume_values": torch.from_numpy(volume_seq.astype(np.float32)),
            "disp_target": disp_target,
            "disp_target_cm": disp_target_cm,
            "disp_time_indices": torch.from_numpy(disp_indices),
            "coord_std": torch.from_numpy(coord_std),
            "static_labels": torch.from_numpy(static_norm.astype(np.float32)),
        }


def compute_stats(mesh_indices: Sequence[int]) -> NormalizationStats:
    coord_sum = np.zeros(3, dtype=np.float64)
    coord_sq_sum = np.zeros(3, dtype=np.float64)
    coord_count = 0
    static_values: List[np.ndarray] = []

    for mesh_idx in mesh_indices:
        mesh_dir = DATA_ROOT / f"mesh{mesh_idx:02d}"
        statics = safe_extract_static_mesh_info(mesh_dir)
        if statics is None:
            continue
        coords = statics.coords
        coord_sum += coords.sum(axis=0)
        coord_sq_sum += np.square(coords).sum(axis=0)
        coord_count += coords.shape[0]
        static_values.append(statics.static_scalars)

    if coord_count == 0:
        raise RuntimeError("No meshes available to compute statistics")

    coord_mean = coord_sum / coord_count
    coord_var = np.maximum(coord_sq_sum / coord_count - np.square(coord_mean), 1e-12)
    coord_std = np.sqrt(coord_var)

    volume_mean = 0.0
    volume_std = 1.0

    static_arr = np.vstack(static_values) if static_values else np.zeros((1, 4), dtype=np.float64)
    if static_arr.size == 0:
        static_mean = np.zeros(4, dtype=np.float32)
        static_std = np.ones(4, dtype=np.float32)
    else:
        static_mean = static_arr.mean(axis=0).astype(np.float32)
        static_std = np.std(static_arr, axis=0).astype(np.float32)
        static_std = np.where(static_std < 1e-6, 1.0, static_std)

    return NormalizationStats(
        coord_mean=coord_mean.astype(np.float32),
        coord_std=coord_std.astype(np.float32),
        volume_mean=volume_mean,
        volume_std=volume_std,
        pressure_scale=float(CONFIG.get("pressure_scale", 1.0)),
        static_mean=static_mean,
        static_std=static_std,
    )


# ---------------------------------------------------------------------------
# Static encoder pretraining dataset helpers
# ---------------------------------------------------------------------------


class StaticEncoderDataset(Dataset):
    def __init__(self, mesh_cache: Dict[int, MeshStatics], stats: NormalizationStats) -> None:
        self.entries: List[Dict[str, torch.Tensor]] = []
        coord_mean = stats.coord_mean.astype(np.float32)
        coord_std = np.where(stats.coord_std < 1e-6, 1.0, stats.coord_std).astype(np.float32)
        static_mean = stats.static_mean.astype(np.float32)
        static_std = np.where(stats.static_std < 1e-6, 1.0, stats.static_std).astype(np.float32)
        for mesh_id, statics in sorted(mesh_cache.items()):
            coords = (statics.coords.astype(np.float32) - coord_mean) / coord_std
            target = (statics.static_scalars.astype(np.float32) - static_mean) / static_std
            self.entries.append(
                {
                    "coords": torch.from_numpy(coords),
                    "edge_index": statics.edge_index.clone(),
                    "target": torch.from_numpy(target),
                    "mesh_id": mesh_id,
                }
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.entries[idx]


def collate_static_batches(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    coords_list: List[torch.Tensor] = []
    batch_idx_list: List[torch.Tensor] = []
    edge_list: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    mesh_ids: List[int] = []
    offset = 0
    for i, item in enumerate(batch):
        coords = item["coords"]
        num_nodes = coords.size(0)
        coords_list.append(coords)
        batch_idx_list.append(torch.full((num_nodes,), i, dtype=torch.long))
        edge_index = item["edge_index"]
        edge_list.append(edge_index + offset)
        targets.append(item["target"])
        mesh_ids.append(int(item.get("mesh_id", i)))
        offset += num_nodes
    coords_tensor = torch.cat(coords_list, dim=0)
    batch_idx = torch.cat(batch_idx_list, dim=0)
    edge_index_tensor = torch.cat(edge_list, dim=1)
    target_tensor = torch.stack(targets, dim=0)
    return {
        "coords": coords_tensor,
        "batch": batch_idx,
        "edge_index": edge_index_tensor,
        "target": target_tensor,
        "mesh_ids": torch.tensor(mesh_ids, dtype=torch.long),
    }


def create_static_encoder(encoder_type: str) -> nn.Module:
    encoder_type = (encoder_type or "mlp").lower()
    if encoder_type == "graph":
        return StaticGraphEncoder()
    return StaticLatentMLP()


def prepare_static_encoder(logger: logging.Logger) -> Tuple[nn.Module, str, Path, bool]:
    requested_type = (CONFIG.get("static_encoder_type") or "mlp").lower()
    weight_path = Path(CONFIG.get("static_encoder_path") or DEFAULT_STATIC_ENCODER_PATH)

    encoder_type = requested_type
    encoder = create_static_encoder(encoder_type)
    loaded = False

    if weight_path.exists():
        state = torch.load(weight_path, map_location="cpu")
        saved_type = (state.get("encoder_type", encoder_type) if isinstance(state, dict) else encoder_type).lower()
        if saved_type != encoder_type:
            logger.warning(
                f"Static encoder type mismatch (requested={encoder_type}, saved={saved_type}); "
                "using saved encoder structure."
            )
            encoder_type = saved_type
            encoder = create_static_encoder(encoder_type)
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        missing = encoder.load_state_dict(state_dict, strict=False)
        missing_keys = getattr(missing, "missing_keys", []) if missing else []
        unexpected = getattr(missing, "unexpected_keys", []) if missing else []
        if missing_keys or unexpected:
            logger.warning(
                f"Static encoder weight load mismatch | missing={missing_keys} unexpected={unexpected}"
            )
        loaded = True
        logger.info(f"Loaded static encoder weights from {weight_path}")
    else:
        if CONFIG.get("freeze_static_head", True):
            raise FileNotFoundError(
                f"Static encoder weights not found at {weight_path}. "
                "Run with --pretrain-static first or disable static head freezing."
            )
        logger.warning(
            f"Static encoder weights not found at {weight_path}; training main model with randomly initialised encoder."
        )

    freeze = bool(CONFIG.get("freeze_static_head", True))
    if freeze:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        logger.info("Static encoder parameters frozen during main training")
    else:
        logger.info("Static encoder parameters will be updated during main training")

    CONFIG["static_encoder_type"] = encoder_type
    CONFIG["static_encoder_path"] = str(weight_path)
    return encoder, encoder_type, weight_path, loaded


# ---------------------------------------------------------------------------
# Graph encoder & decoders
# ---------------------------------------------------------------------------


class ResidualGraphBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm = GraphNorm(in_channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(DROPOUT_RATE)
        self.conv = GATv2Conv(
            in_channels,
            out_channels,
            heads=GAT_HEADS,
            concat=False,
            dropout=DROPOUT_RATE,
        )
        self.proj = None if in_channels == out_channels else nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.norm(x, batch)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv(h, edge_index)
        residual = x if self.proj is None else self.proj(x)
        return residual + h


class GlobalFusionBlock(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.local_proj = nn.Linear(feature_dim, feature_dim)
        self.global_proj = nn.Linear(feature_dim, feature_dim)
        self.attn = nn.MultiheadAttention(feature_dim, GLOBAL_FUSION_HEADS, dropout=DROPOUT_RATE, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        if features.numel() == 0:
            return features
        global_feat = global_mean_pool(features, batch_idx)
        local_proj = self.local_proj(features)
        global_proj = self.global_proj(global_feat)[batch_idx]
        attn_out, _ = self.attn(local_proj.unsqueeze(0), global_proj.unsqueeze(0), global_proj.unsqueeze(0))
        return self.norm(local_proj + attn_out.squeeze(0))


class GraphFusionEncoder(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.block1a = ResidualGraphBlock(in_channels, HIDDEN_CHANNELS)
        self.block1b = ResidualGraphBlock(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.block1c = ResidualGraphBlock(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.fusion1 = GlobalFusionBlock(HIDDEN_CHANNELS)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        stages: List[torch.Tensor] = []
        batches: List[torch.Tensor] = []

        x = self.block1a(x, edge_index, batch_idx)
        x = self.block1b(x, edge_index, batch_idx)
        x = self.block1c(x, edge_index, batch_idx)
        stages.append(x)
        batches.append(batch_idx)

        fused = self.fusion1(x, batch_idx)
        stages.append(fused)
        batches.append(batch_idx)

        return fused, stages, batches


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(DROPOUT_RATE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = self.drop(h)
        residual = h
        h = self.act(self.fc2(h))
        h = self.drop(h)
        h = h + residual
        return self.fc3(h)


class StaticLatentMLP(nn.Module):
    def __init__(self, feature_dim: int = 3, hidden_dim: int = HIDDEN_CHANNELS) -> None:
        super().__init__()
        self.coord_proj = nn.Linear(feature_dim, hidden_dim)
        self.fusion = GlobalFusionBlock(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(
        self,
        coords: torch.Tensor,
        batch_idx: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = self.coord_proj(coords)
        fused = self.fusion(feat, batch_idx)
        pooled = global_mean_pool(fused, batch_idx)
        return self.head(pooled)


class StaticGraphEncoder(nn.Module):
    def __init__(self, feature_dim: int = 3, hidden_dim: int = HIDDEN_CHANNELS) -> None:
        super().__init__()
        self.encoder = GraphFusionEncoder(feature_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(
        self,
        coords: torch.Tensor,
        batch_idx: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_index is None:
            raise ValueError("edge_index is required for graph static encoder")
        fused, _, _ = self.encoder(coords, edge_index, batch_idx)
        pooled = global_mean_pool(fused, batch_idx)
        return self.head(pooled)


class CGFENet(nn.Module):
    def __init__(
        self,
        node_in: int,
        dynamic_dim: int,
        hidden: int = 128,
        apply_sigmoid: bool = False,
        fusion_mode: str = "layernorm",
        static_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.apply_sigmoid = bool(apply_sigmoid)
        self.fusion_mode = fusion_mode
        self.encoder = GraphFusionEncoder(node_in)

        shallow_dim = HIDDEN_CHANNELS
        combined_dim = shallow_dim * 2
        self.mesh_proj = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, hidden),
            nn.ReLU(),
        )
        self.node_proj = nn.Sequential(
            nn.Linear(shallow_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.dynamic_mlp = nn.Sequential(
            nn.Linear(dynamic_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, hidden),
            nn.ReLU(),
        )
        self.temporal_gru = nn.GRU(hidden, hidden, batch_first=True)
        if self.fusion_mode == "layernorm":
            self.context_fusion = nn.LayerNorm(hidden)
        else:
            self.context_fusion = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )

        self.init_state_mlp = nn.Sequential(
            nn.Linear(dynamic_dim, hidden),
            nn.Tanh(),
        )

        pressure_layers: List[nn.Module] = [
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        ]
        if self.apply_sigmoid:
            pressure_layers.append(nn.Sigmoid())
        self.pressure_head = nn.Sequential(*pressure_layers)

        self.temporal_proj = nn.Linear(hidden, hidden)
        self.disp_head = ResidualMLP(hidden, hidden, 3)

        self.static_head = static_encoder if static_encoder is not None else StaticLatentMLP()
        self.static_head_frozen = not any(param.requires_grad for param in self.static_head.parameters())

        self.inverse_mesh_proj = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, hidden),
            nn.ReLU(),
        )
        self.inverse_node_proj = nn.Sequential(
            nn.Linear(shallow_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.inverse_temporal_proj = nn.Linear(hidden, hidden)
        self.inverse_disp_head = ResidualMLP(hidden, hidden, 3)

        self._init_weights()

    def train(self, mode: bool = True) -> "CGFENet":
        super().train(mode)
        if self.static_head_frozen:
            self.static_head.eval()
        return self

    def _init_weights(self) -> None:
        modules = [
            self.mesh_proj,
            self.node_proj,
            self.dynamic_mlp,
            self.pressure_head,
            self.init_state_mlp,
            self.temporal_proj,
            self.disp_head,
            self.inverse_mesh_proj,
            self.inverse_node_proj,
            self.inverse_temporal_proj,
            self.inverse_disp_head,
            self.static_head,
        ]
        if isinstance(self.context_fusion, nn.Sequential):
            modules.append(self.context_fusion)
        for module in modules:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _decode_branch(
        self,
        stages: List[torch.Tensor],
        batches: List[torch.Tensor],
        mesh_proj: nn.Module,
        node_proj: nn.Module,
        temporal_proj: nn.Module,
        pressure_head: Optional[nn.Module],
        disp_head: nn.Module,
        gru_out: torch.Tensor,
        batch_idx: torch.Tensor,
        disp_indices: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        combined_features = torch.cat([stages[0], stages[1]], dim=-1)
        mesh_context = mesh_proj(global_mean_pool(combined_features, batches[0]))
        fusion_input = mesh_context.unsqueeze(1) + gru_out
        fused_context = self.context_fusion(fusion_input)
        pressure_pred: Optional[torch.Tensor] = None
        if pressure_head is not None:
            pressure_pred = pressure_head(fused_context).squeeze(-1)

        temporal_context = temporal_proj(fused_context)
        if disp_indices is not None:
            temporal_context = temporal_context.index_select(1, disp_indices)

        node_features = node_proj(stages[1])
        temporal_for_nodes = temporal_context[batch_idx]

        node_expanded = node_features.unsqueeze(1).expand(-1, temporal_for_nodes.size(1), -1)
        fused_disp = node_expanded + temporal_for_nodes

        disp_pred = disp_head(fused_disp.view(-1, fused_disp.size(-1))).view(
            node_features.size(0),
            temporal_for_nodes.size(1),
            3,
        )
        return pressure_pred, disp_pred

    @staticmethod
    def _expand_latent(latent: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
        counts = (ptr[1:] - ptr[:-1]).to(latent.device, dtype=torch.long)
        return torch.repeat_interleave(latent, counts, dim=0)

    def forward(
        self,
        graph_batch: Data,
        dynamic_feats: torch.Tensor,
        disp_indices: Optional[torch.Tensor] = None,
        real_disp: Optional[torch.Tensor] = None,
        static_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        batch_idx = graph_batch.batch
        ptr = graph_batch.ptr
        edge_index = graph_batch.edge_index

        dyn_embed = self.dynamic_mlp(dynamic_feats)
        first_step = dynamic_feats[:, 0, :]
        init_state = self.init_state_mlp(first_step).unsqueeze(0)
        gru_out, _ = self.temporal_gru(dyn_embed, init_state.contiguous())
        if disp_indices is not None and disp_indices.dim() == 0:
            disp_indices = disp_indices.unsqueeze(0)

        bottleneck_fwd, stages_fwd, batches_fwd = self.encoder(
            graph_batch.x,
            graph_batch.edge_index,
            batch_idx,
        )
        pressure_fwd, disp_fwd = self._decode_branch(
            stages_fwd,
            batches_fwd,
            self.mesh_proj,
            self.node_proj,
            self.temporal_proj,
            self.pressure_head,
            self.disp_head,
            gru_out,
            batch_idx,
            disp_indices,
        )

        coords_norm = graph_batch.x[:, :3]
        warped_coords = coords_norm + disp_fwd[:, -1, :]
        z_loaded = self.static_head(warped_coords, batch_idx, edge_index)
        static_expanded_loaded = self._expand_latent(z_loaded, ptr)
        region_flag = graph_batch.x[:, 3:4]
        inverse_input = torch.cat([warped_coords, region_flag, static_expanded_loaded], dim=-1)

        bottleneck_inv, stages_inv, batches_inv = self.encoder(
            inverse_input,
            graph_batch.edge_index,
            batch_idx,
        )
        _, disp_inv = self._decode_branch(
            stages_inv,
            batches_inv,
            self.inverse_mesh_proj,
            self.inverse_node_proj,
            self.inverse_temporal_proj,
            None,
            self.inverse_disp_head,
            gru_out,
            batch_idx,
            disp_indices,
        )
        inverse_mesh = warped_coords + disp_inv[:, -1, :]
        z_unloaded = self.static_head(inverse_mesh, batch_idx, edge_index)
        static_expanded_unloaded = self._expand_latent(z_unloaded, ptr)

        cycle_input = torch.cat([inverse_mesh, region_flag, static_expanded_unloaded], dim=-1)
        bottleneck_cycle, stages_cycle, batches_cycle = self.encoder(
            cycle_input,
            graph_batch.edge_index,
            batch_idx,
        )
        pressure_cycle, disp_cycle = self._decode_branch(
            stages_cycle,
            batches_cycle,
            self.mesh_proj,
            self.node_proj,
            self.temporal_proj,
            self.pressure_head,
            self.disp_head,
            gru_out,
            batch_idx,
            disp_indices,
        )
        cycle_mesh = inverse_mesh + disp_cycle[:, -1, :]
        z_hat_loaded = self.static_head(cycle_mesh, batch_idx, edge_index)

        inverse_sup = {}
        if real_disp is not None:
            real_disp = real_disp.to(coords_norm.device)
            real_loaded = coords_norm + real_disp[:, -1, :]
            if static_labels is not None:
                static_labels = static_labels.to(coords_norm.device)
                expanded_static = self._expand_latent(static_labels, ptr)
            else:
                expanded_static = self._expand_latent(z_loaded.detach(), ptr)
            inverse_sup_input = torch.cat([real_loaded, region_flag, expanded_static], dim=-1)
            _, stages_sup, batches_sup = self.encoder(
                inverse_sup_input,
                graph_batch.edge_index,
                batch_idx,
            )
            _, disp_sup = self._decode_branch(
                stages_sup,
                batches_sup,
                self.inverse_mesh_proj,
                self.inverse_node_proj,
                self.inverse_temporal_proj,
                None,
                self.inverse_disp_head,
                gru_out,
                batch_idx,
                disp_indices,
            )
            sup_mesh = real_loaded + disp_sup[:, -1, :]
            inverse_sup = {
                "disp": disp_sup,
                "mesh": sup_mesh,
                "real_loaded": real_loaded,
            }

        return {
            "forward": {
                "pressure": pressure_fwd,
                "disp": disp_fwd,
                "mesh": warped_coords,
                "z_loaded": z_loaded,
            },
            "cycle1": {
                "disp": disp_inv,
                "mesh": inverse_mesh,
                "z_unloaded": z_unloaded,
            },
            "cycle2": {
                "pressure": pressure_cycle,
                "disp": disp_cycle,
                "mesh": cycle_mesh,
                "z_hat": z_hat_loaded,
            },
            "inverse_supervised": inverse_sup,
            "disp_time_indices": disp_indices,
        }


# ---------------------------------------------------------------------------
# Volume utilities
def compute_losses(
    outputs: Dict[str, Dict[str, torch.Tensor]],
    batch: Dict[str, torch.Tensor],
    config: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    forward_out = outputs["forward"]
    cycle1_out = outputs["cycle1"]
    cycle2_out = outputs["cycle2"]
    inverse_sup_out = outputs.get("inverse_supervised") or {}
    disp_indices = outputs.get("disp_time_indices")

    device = forward_out["pressure"].device

    pressure_target = batch["pressure_target"].to(device)
    disp_target_full = batch["disp_target"].to(device)
    static_labels = batch["static_labels"].to(device)
    graph_batch: Batch = batch["graph_batch"]

    if disp_indices is not None and disp_target_full.size(1) != forward_out["disp"].size(1):
        disp_target = disp_target_full.index_select(1, disp_indices.to(device))
    else:
        disp_target = disp_target_full

    scale = float(config.get("pressure_scale", 1.0)) if config.get("normalize_pressure", False) else 1.0

    pressure_loss = F.mse_loss(forward_out["pressure"], pressure_target)
    disp_loss = F.mse_loss(forward_out["disp"], disp_target)

    cycle2_pressure_loss = F.mse_loss(cycle2_out["pressure"], pressure_target)

    base_coords = graph_batch.x[:, :3].to(device)
    cycle1_mesh_loss = F.mse_loss(cycle1_out["mesh"], base_coords)
    target_loaded = base_coords
    if disp_target.size(1) > 0:
        target_loaded = base_coords + disp_target[:, -1, :]
    cycle2_mesh_loss = F.mse_loss(cycle2_out["mesh"], target_loaded)

    inverse_sup_mesh_loss = torch.tensor(0.0, device=device)
    if inverse_sup_out:
        inverse_sup_mesh_loss = F.mse_loss(inverse_sup_out["mesh"], base_coords)

    static_recon_loss = F.mse_loss(cycle1_out["z_unloaded"], static_labels)
    latent_cycle_loss = F.mse_loss(cycle2_out["z_hat"], forward_out["z_loaded"].detach())

    total_loss = (
        config.get("pressure_loss_weight", 1.0)
        * (pressure_loss + cycle2_pressure_loss)
        + config.get("disp_loss_weight", 1.0)
        * (disp_loss + cycle1_mesh_loss + cycle2_mesh_loss + inverse_sup_mesh_loss)
    )

    diff_forward = (forward_out["pressure"] - pressure_target) * scale
    forward_pressure_rmse = torch.sqrt(torch.mean(diff_forward.pow(2), dim=1) + 1e-12).mean()

    diff_cycle2 = (cycle2_out["pressure"] - pressure_target) * scale
    cycle2_pressure_rmse = torch.sqrt(torch.mean(diff_cycle2.pow(2), dim=1) + 1e-12).mean()

    coord_std = batch.get("coord_std")
    if coord_std is not None:
        coord_std_tensor = coord_std.to(device).view(1, 1, 3)
        coord_std_vec = coord_std.to(device).view(1, 3)
    else:
        ones = torch.ones(3, device=device, dtype=forward_out["disp"].dtype)
        coord_std_tensor = ones.view(1, 1, 3)
        coord_std_vec = ones.view(1, 3)

    if "disp_target_cm" in batch:
        disp_target_cm_full = batch["disp_target_cm"].to(device)
        if disp_indices is not None and disp_target_cm_full.size(1) != forward_out["disp"].size(1):
            disp_target_cm = disp_target_cm_full.index_select(1, disp_indices.to(device))
        else:
            disp_target_cm = disp_target_cm_full
    else:
        disp_target_cm = disp_target * coord_std_tensor

    disp_pred_cm = forward_out["disp"] * coord_std_tensor
    disp_error_cm = disp_pred_cm - disp_target_cm
    disp_rmse = torch.sqrt(torch.mean(disp_error_cm.pow(2)) + 1e-12)
    disp_error_norm = torch.norm(disp_error_cm, dim=-1)
    disp_overlap = compute_overlap(disp_error_norm, threshold_cm=0.1)
    disp_overlap_0p2 = compute_overlap(disp_error_norm, threshold_cm=0.2)

    cycle1_error_cm = (cycle1_out["mesh"] - base_coords) * coord_std_vec
    cycle1_disp_rmse = torch.sqrt(torch.mean(cycle1_error_cm.pow(2)) + 1e-12)
    cycle1_error_norm = torch.norm(cycle1_error_cm, dim=-1, keepdim=False)
    cycle1_disp_overlap = compute_overlap(cycle1_error_norm, threshold_cm=0.1)
    cycle1_disp_overlap_0p2 = compute_overlap(cycle1_error_norm, threshold_cm=0.2)

    cycle2_error_cm = (cycle2_out["mesh"] - target_loaded) * coord_std_vec
    cycle2_disp_rmse = torch.sqrt(torch.mean(cycle2_error_cm.pow(2)) + 1e-12)
    cycle2_error_norm = torch.norm(cycle2_error_cm, dim=-1, keepdim=False)
    cycle2_disp_overlap = compute_overlap(cycle2_error_norm, threshold_cm=0.1)
    cycle2_disp_overlap_0p2 = compute_overlap(cycle2_error_norm, threshold_cm=0.2)

    inverse_sup_disp_rmse = torch.tensor(0.0, device=device)
    inverse_sup_disp_overlap = torch.tensor(0.0, device=device)
    inverse_sup_disp_overlap_0p2 = torch.tensor(0.0, device=device)
    if inverse_sup_out:
        inverse_sup_error_cm = (inverse_sup_out["mesh"] - base_coords) * coord_std_vec
        inverse_sup_disp_rmse = torch.sqrt(torch.mean(inverse_sup_error_cm.pow(2)) + 1e-12)
        inverse_sup_error_norm = torch.norm(inverse_sup_error_cm, dim=-1, keepdim=False)
        inverse_sup_disp_overlap = compute_overlap(inverse_sup_error_norm, threshold_cm=0.1)
        inverse_sup_disp_overlap_0p2 = compute_overlap(inverse_sup_error_norm, threshold_cm=0.2)

    static_diff_norm = cycle1_out["z_unloaded"] - static_labels
    static_rmse_norm = torch.sqrt(torch.mean(static_diff_norm.pow(2)) + 1e-12)
    static_mean_cfg = CONFIG.get("static_mean")
    static_std_cfg = CONFIG.get("static_std")
    if static_mean_cfg is not None and static_std_cfg is not None:
        static_mean_tensor = torch.as_tensor(static_mean_cfg, device=device, dtype=cycle1_out["z_unloaded"].dtype)
        static_std_tensor = torch.as_tensor(static_std_cfg, device=device, dtype=cycle1_out["z_unloaded"].dtype)
    else:
        static_mean_tensor = torch.zeros(static_labels.size(-1), device=device, dtype=cycle1_out["z_unloaded"].dtype)
        static_std_tensor = torch.ones(static_labels.size(-1), device=device, dtype=cycle1_out["z_unloaded"].dtype)
    pred_static_physical = cycle1_out["z_unloaded"] * static_std_tensor + static_mean_tensor
    target_static_physical = static_labels * static_std_tensor + static_mean_tensor
    static_rmse_physical = torch.sqrt(torch.mean((pred_static_physical - target_static_physical).pow(2)) + 1e-12)
    latent_cycle_rmse = torch.sqrt(torch.mean((cycle2_out["z_hat"] - forward_out["z_loaded"]).pow(2)) + 1e-12)

    # Forward pressure correlation and R^2 (cheap to compute, per-sequence over time)
    # Compute Pearson correlation per batch item and average; R^2 is its square
    with torch.no_grad():
        pred_center = forward_out["pressure"] - forward_out["pressure"].mean(dim=1, keepdim=True)
        target_center = pressure_target - pressure_target.mean(dim=1, keepdim=True)
        denom = torch.sqrt(
            torch.mean(pred_center.pow(2), dim=1) * torch.mean(target_center.pow(2), dim=1) + 1e-12
        )
        corr = torch.where(
            denom > 1e-8,
            torch.mean(pred_center * target_center, dim=1) / denom,
            torch.zeros_like(denom),
        )
        pressure_r = corr.mean()
        pressure_r2 = (corr.pow(2)).mean()

    forward_mesh_mm = forward_out["mesh"] * coord_std_vec
    target_loaded_mm = target_loaded * coord_std_vec
    cycle1_mesh_mm = cycle1_out["mesh"] * coord_std_vec
    base_coords_mm = base_coords * coord_std_vec
    cycle2_mesh_mm = cycle2_out["mesh"] * coord_std_vec

    forward_hd = max(
        hausdorff_directional(forward_mesh_mm, target_loaded_mm),
        hausdorff_directional(target_loaded_mm, forward_mesh_mm),
    )
    cycle1_hd = max(
        hausdorff_directional(cycle1_mesh_mm, base_coords_mm),
        hausdorff_directional(base_coords_mm, cycle1_mesh_mm),
    )
    cycle2_hd = max(
        hausdorff_directional(cycle2_mesh_mm, target_loaded_mm),
        hausdorff_directional(target_loaded_mm, cycle2_mesh_mm),
    )

    inverse_sup_hd = torch.tensor(0.0, device=device)
    if inverse_sup_out:
        inverse_sup_mesh_mm = inverse_sup_out["mesh"] * coord_std_vec
        inverse_sup_hd = max(
            hausdorff_directional(inverse_sup_mesh_mm, base_coords_mm),
            hausdorff_directional(base_coords_mm, inverse_sup_mesh_mm),
        )

    metrics = {
        "loss": float(total_loss.item()),
        "pressure_loss": float(pressure_loss.item()),
        "cycle2_pressure_loss": float(cycle2_pressure_loss.item()),
        "disp_loss": float(disp_loss.item()),
        "cycle1_mesh_loss": float(cycle1_mesh_loss.item()),
        "cycle2_mesh_loss": float(cycle2_mesh_loss.item()),
        "inverse_supervised_mesh_loss": float(inverse_sup_mesh_loss.item()),
        "static_recon_loss": float(static_recon_loss.item()),
        "latent_cycle_loss": float(latent_cycle_loss.item()),
        "pressure_rmse": float(forward_pressure_rmse.item()),
        "cycle2_pressure_rmse": float(cycle2_pressure_rmse.item()),
        "pressure_r": float(pressure_r.item()),
        "pressure_r2": float(pressure_r2.item()),
        "disp_rmse": float(disp_rmse.item()),
        "cycle1_disp_rmse": float(cycle1_disp_rmse.item()),
        "cycle2_disp_rmse": float(cycle2_disp_rmse.item()),
        "inverse_supervised_disp_rmse": float(inverse_sup_disp_rmse.item()),
        "disp_overlap": float(disp_overlap),
        "disp_overlap_0p2": float(disp_overlap_0p2),
        "cycle1_disp_overlap": float(cycle1_disp_overlap),
        "cycle1_disp_overlap_0p2": float(cycle1_disp_overlap_0p2),
        "cycle2_disp_overlap": float(cycle2_disp_overlap),
        "cycle2_disp_overlap_0p2": float(cycle2_disp_overlap_0p2),
        "inverse_supervised_disp_overlap": float(inverse_sup_disp_overlap),
        "inverse_supervised_disp_overlap_0p2": float(inverse_sup_disp_overlap_0p2),
        "static_rmse_norm": float(static_rmse_norm.item()),
        "static_rmse": float(static_rmse_physical.item()),
        "latent_cycle_rmse": float(latent_cycle_rmse.item()),
        "forward_hd": float(forward_hd),
        "cycle1_hd": float(cycle1_hd),
        "cycle2_hd": float(cycle2_hd),
        "inverse_supervised_hd": float(inverse_sup_hd),
    }

    return total_loss, metrics


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def compute_overlap(disp_error: torch.Tensor, threshold_cm: float = DISP_OVERLAP_THRESHOLD_CM) -> float:
    return (disp_error <= threshold_cm).float().mean().item()


def hausdorff_directional(source: torch.Tensor, target: torch.Tensor, chunk: int = 4096) -> float:
    if source.numel() == 0 or target.numel() == 0:
        return 0.0
    # Use smaller configurable chunk to reduce memory usage
    chunk = int(CONFIG.get("hd_chunk", 1024))
    max_min = float("-inf")
    # Compute on CPU to reduce GPU memory pressure
    src_cpu = source.detach().to("cpu")
    tgt_cpu = target.detach().to("cpu")
    for start in range(0, src_cpu.size(0), chunk):
        end = start + chunk
        dist_block = torch.cdist(src_cpu[start:end], tgt_cpu)
        block_max = dist_block.min(dim=1).values.max().item()
        if block_max > max_min:
            max_min = block_max
    if max_min == float("-inf"):
        return 0.0
    return max_min


def accumulate_summary_metrics(
    metrics: Dict[str, List[float]],
    outputs: Dict[str, Dict[str, torch.Tensor]],
    graph_batch: Batch,
    pressure_target: torch.Tensor,
    disp_target: torch.Tensor,
    disp_target_cm: Optional[torch.Tensor],
    coord_std: Optional[torch.Tensor],
    disp_indices: Optional[torch.Tensor],
    pressure_scale: float,
) -> None:
    device = pressure_target.device
    coord_std_tensor = (
        coord_std.view(1, 1, 3)
        if coord_std is not None
        else torch.ones(1, 1, 3, device=device, dtype=pressure_target.dtype)
    )
    coord_std_vec = coord_std_tensor.view(1, 3)

    if disp_target_cm is None:
        disp_target_use = disp_target
        if disp_indices is not None and disp_target_use.size(1) != outputs["forward"]["disp"].size(1):
            disp_target_use = disp_target_use.index_select(1, disp_indices)
        disp_target_cm_use = disp_target_use * coord_std_tensor
    else:
        disp_target_cm_use = disp_target_cm
        if disp_indices is not None and disp_target_cm_use.size(1) != outputs["forward"]["disp"].size(1):
            disp_target_cm_use = disp_target_cm_use.index_select(1, disp_indices)

    forward_out = outputs["forward"]
    forward_pressure = forward_out["pressure"] * pressure_scale
    forward_disp = forward_out["disp"]
    if disp_indices is not None and forward_disp.size(1) != disp_target_cm_use.size(1):
        forward_disp = forward_disp.index_select(1, disp_indices)
    forward_disp_cm = forward_disp * coord_std_tensor

    inverse_out = outputs.get("inverse_supervised")
    inverse_disp_cm = None
    if inverse_out:
        inverse_disp = inverse_out["disp"]
        if disp_indices is not None and inverse_disp.size(1) != disp_target_cm_use.size(1):
            inverse_disp = inverse_disp.index_select(1, disp_indices)
        inverse_disp_cm = inverse_disp * coord_std_tensor

    def _collect_pressure(prefix: str, pred: torch.Tensor, target: torch.Tensor) -> None:
        error = pred - target
        rmse = torch.sqrt(torch.mean(error.pow(2), dim=1) + 1e-12)
        pred_center = pred - pred.mean(dim=1, keepdim=True)
        target_center = target - target.mean(dim=1, keepdim=True)
        denom = torch.sqrt(
            torch.mean(pred_center.pow(2), dim=1) * torch.mean(target_center.pow(2), dim=1) + 1e-12
        )
        corr = torch.where(
            denom > 1e-8,
            torch.mean(pred_center * target_center, dim=1) / denom,
            torch.zeros_like(denom),
        )
        metrics[f"{prefix}_pressure_rmse"].extend(rmse.detach().cpu().tolist())
        metrics[f"{prefix}_pressure_r"].extend(corr.detach().cpu().tolist())

    pressure_target_scaled = pressure_target * pressure_scale
    _collect_pressure("forward", forward_pressure, pressure_target_scaled)

    ptr = graph_batch.ptr
    coords_norm_all = graph_batch.x[:, :3].to(device)
    for graph_idx in range(graph_batch.num_graphs):
        start = int(ptr[graph_idx].item())
        end = int(ptr[graph_idx + 1].item())

        coord_sample = coords_norm_all[start:end]
        coord_sample_cm = coord_sample.unsqueeze(1) * coord_std_tensor

        forward_pred_sample = forward_disp_cm[start:end]
        target_sample = disp_target_cm_use[start:end]

        forward_error = forward_pred_sample - target_sample
        forward_error_norm = torch.norm(forward_error, dim=-1)
        metrics["forward_disp_rmse"].append(torch.sqrt(torch.mean(forward_error.pow(2)) + 1e-12).item())
        metrics["forward_disp_overlap_0p1"].append(compute_overlap(forward_error_norm, 0.1))
        metrics["forward_disp_overlap_0p2"].append(compute_overlap(forward_error_norm, 0.2))

        forward_mesh = coord_sample_cm + forward_pred_sample
        target_mesh = coord_sample_cm + target_sample
        forward_points = forward_mesh.reshape(-1, 3)
        target_points = target_mesh.reshape(-1, 3)
        hd_forward = max(
            hausdorff_directional(forward_points, target_points),
            hausdorff_directional(target_points, forward_points),
        )
        metrics["forward_disp_hd"].append(hd_forward)

        if inverse_disp_cm is not None:
            inverse_pred_sample = inverse_disp_cm[start:end]
            inverse_target_sample = -target_sample
            inverse_error = inverse_pred_sample - inverse_target_sample
            inverse_error_norm = torch.norm(inverse_error, dim=-1)
            metrics["inverse_disp_rmse"].append(
                torch.sqrt(torch.mean(inverse_error.pow(2)) + 1e-12).item()
            )
            # Displacement-space overlaps (less consistent with training losses)
            metrics["inverse_disp_overlap_0p1"].append(compute_overlap(inverse_error_norm, 0.1))
            metrics["inverse_disp_overlap_0p2"].append(compute_overlap(inverse_error_norm, 0.2))

            loaded_reference = coord_sample_cm + target_sample
            if inverse_pred_sample.size(1) != loaded_reference.size(1):
                if disp_indices is not None:
                    loaded_reference = loaded_reference.index_select(1, disp_indices)
                else:
                    loaded_reference = loaded_reference[:, : inverse_pred_sample.size(1), :]

            inverse_target_mesh = loaded_reference + inverse_target_sample
            inverse_mesh = loaded_reference + inverse_pred_sample
            base_mesh = coord_sample_cm.expand(-1, inverse_mesh.size(1), -1)
            inverse_mesh_final = inverse_mesh[:, -1, :]
            inverse_target_final = inverse_target_mesh[:, -1, :]
            base_mesh_final = base_mesh[:, -1, :]

            inverse_points = inverse_mesh_final.reshape(-1, 3)
            inverse_target_points = inverse_target_final.reshape(-1, 3)
            hd_inverse = max(
                hausdorff_directional(inverse_points, inverse_target_points),
                hausdorff_directional(inverse_target_points, inverse_points),
            )
            metrics["inverse_disp_hd"].append(hd_inverse)

            mesh_err = inverse_mesh_final - base_mesh_final
            metrics["inverse_mesh_rmse"].append(torch.sqrt(torch.mean(mesh_err.pow(2)) + 1e-12).item())
            mesh_err_norm = torch.norm(mesh_err, dim=-1)
            metrics["inverse_mesh_overlap_0p1"].append(compute_overlap(mesh_err_norm, 0.1))
            metrics["inverse_mesh_overlap_0p2"].append(compute_overlap(mesh_err_norm, 0.2))
            hd_mesh = max(
                hausdorff_directional(inverse_points, base_mesh_final.reshape(-1, 3)),
                hausdorff_directional(base_mesh_final.reshape(-1, 3), inverse_points),
            )
            metrics["inverse_mesh_hd"].append(hd_mesh)


def summarise_metrics(metrics: Dict[str, List[float]]) -> Dict[str, Tuple[float, float]]:
    def stats(values: List[float]) -> Tuple[float, float]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return 0.0, 0.0
        return float(arr.mean()), float(arr.std(ddof=0))

    return {key: stats(vals) for key, vals in metrics.items()}


def log_detailed_metrics(
    logger: logging.Logger,
    epoch: int,
    tag: str,
    model: nn.Module,
    loader: DataLoader,
    config: Dict[str, float],
) -> None:
    _, _, metric_lists = evaluate(model, loader, config, collect_summary=True)
    summary = summarise_metrics(metric_lists or {})
    logger.info(
        f"Epoch {epoch:03d} | {tag} detailed -> "
        f"Fwd disp rmse={summary.get('forward_disp_rmse', (0.0, 0.0))[0]:.4f} | "
        f"Fwd overlap@0.1={summary.get('forward_disp_overlap_0p1', (0.0, 0.0))[0]:.4f} | "
        f"Fwd overlap@0.2={summary.get('forward_disp_overlap_0p2', (0.0, 0.0))[0]:.4f} | "
        f"Fwd HD={summary.get('forward_disp_hd', (0.0, 0.0))[0]:.4f} | "
        f"Fwd P_RMSE={summary.get('forward_pressure_rmse', (0.0, 0.0))[0]:.4f} | "
        f"Fwd P_r={summary.get('forward_pressure_r', (0.0, 0.0))[0]:.4f} || "
        f"Inv disp rmse={summary.get('inverse_disp_rmse', (0.0, 0.0))[0]:.4f} | "
        f"Inv overlap@0.1={summary.get('inverse_disp_overlap_0p1', (0.0, 0.0))[0]:.4f} | "
        f"Inv overlap@0.2={summary.get('inverse_disp_overlap_0p2', (0.0, 0.0))[0]:.4f} | "
        f"Inv HD={summary.get('inverse_disp_hd', (0.0, 0.0))[0]:.4f}"
    )


def _square_stats(mean: float, std: float) -> Tuple[float, float]:
    var = std ** 2
    mean_sq = mean ** 2
    var_sq = 4 * mean ** 2 * var
    std_sq = np.sqrt(var_sq)
    return mean_sq, std_sq


def format_summary_lines(summary: Dict[str, Tuple[float, float]]) -> List[str]:
    lines: List[str] = []
    for label, key, transform in SUMMARY_SECTIONS:
        mean, std = summary.get(key, (0.0, 0.0))
        if transform == "square":
            mean, std = _square_stats(mean, std)
        lines.append(f"{label}: {mean:.6f} ± {std:.6f}")
    return lines


def save_summary_to_file(summary: Dict[str, Tuple[float, float]], path: Path) -> None:
    lines = format_summary_lines(summary)
    with open(path, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Collate & helpers
# ---------------------------------------------------------------------------


class RunningStats:
    __slots__ = ("count", "mean", "M2")

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        return max(self.M2 / (self.count - 1), 0.0)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)



def collate_sequences(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    graphs = [item["graph"] for item in batch]
    graph_batch = Batch.from_data_list(graphs)

    dynamic = torch.stack([item["dynamic"] for item in batch], dim=0)
    pressure_target = torch.stack([item["pressure_target"] for item in batch], dim=0)
    mesh_indices = torch.tensor([item["mesh_idx"] for item in batch], dtype=torch.long)
    time_values = torch.stack([item["time_values"] for item in batch], dim=0)
    volume_values = torch.stack([item["volume_values"] for item in batch], dim=0)
    disp_target = torch.cat([item["disp_target"] for item in batch], dim=0)
    static_labels = torch.stack([item["static_labels"] for item in batch], dim=0)

    result: Dict[str, torch.Tensor] = {
        "graph_batch": graph_batch,
        "dynamic": dynamic,
        "pressure_target": pressure_target,
        "disp_target": disp_target,
        "mesh_indices": mesh_indices,
        "times": time_values,
        "volume_values": volume_values,
        "static_labels": static_labels,
    }

    if batch[0].get("disp_target_cm") is not None:
        disp_target_cm = torch.cat(
            [item["disp_target_cm"] for item in batch if item.get("disp_target_cm") is not None],
            dim=0,
        )
        result["disp_target_cm"] = disp_target_cm

    disp_indices = batch[0].get("disp_time_indices")
    if disp_indices is not None:
        result["disp_time_indices"] = disp_indices

    if "coord_std" in batch[0]:
        result["coord_std"] = batch[0]["coord_std"]

    return result


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------


def plot_epi_endo_overview(mesh_cache: Dict[int, MeshStatics], output_dir: Path) -> None:
    if not mesh_cache:
        return
    num_meshes = len(mesh_cache)
    cols = min(4, num_meshes)
    rows = math.ceil(num_meshes / cols)
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    for idx, (mesh_id, statics) in enumerate(sorted(mesh_cache.items())):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        if statics.endo_faces.size > 0:
            endo_poly = Poly3DCollection(
                statics.coords[statics.endo_faces],
                facecolor=(0.9, 0.1, 0.1, 0.65),
                edgecolor=(0.6, 0.0, 0.0, 0.9),
                linewidths=0.3,
            )
            ax.add_collection3d(endo_poly)
        endo_pts = statics.coords[statics.endo_mask > 0.5]
        if endo_pts.size:
            ax.scatter(endo_pts[:, 0], endo_pts[:, 1], endo_pts[:, 2], s=2, color="darkred")

        if statics.epi_faces.size > 0:
            epi_poly = Poly3DCollection(
                statics.coords[statics.epi_faces],
                facecolor=(0.2, 0.4, 1.0, 0.35),
                edgecolor=(0.1, 0.2, 0.6, 0.5),
                linewidths=0.2,
            )
            ax.add_collection3d(epi_poly)
        epi_pts = statics.coords[statics.epi_mask > 0.5]
        if epi_pts.size:
            ax.scatter(epi_pts[:, 0], epi_pts[:, 1], epi_pts[:, 2], s=1, color="navy")

        ax.auto_scale_xyz(statics.coords[:, 0], statics.coords[:, 1], statics.coords[:, 2])
        try:
            ax.set_box_aspect((1.0, 1.0, 1.0))
        except AttributeError:
            pass
        ax.view_init(elev=25.0, azim=-60.0)
        ax.set_title(f"mesh {mesh_id:02d}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        if idx == 0:
            proxy_endo = plt.Line2D([0], [0], color="darkred", marker="s", linestyle="")
            proxy_epi = plt.Line2D([0], [0], color="navy", marker="s", linestyle="")
            ax.legend([proxy_endo, proxy_epi], ["endo", "epi"], loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "epi_endo_overview.png", dpi=200)
    plt.close(fig)


def plot_pressure_snapshot(
    model: CGFENet,
    dataset: FixedSequenceDataset,
    mesh_idx: int,
    output_dir: Path,
    title_prefix: str,
) -> None:
    sample_indices = [i for i, (m_idx, _) in enumerate(dataset.samples) if m_idx == mesh_idx]
    if not sample_indices:
        available_meshes = sorted({m_idx for m_idx, _ in dataset.samples})
        if not available_meshes:
            return
        mesh_idx = available_meshes[0]
        sample_indices = [i for i, (m_idx, _) in enumerate(dataset.samples) if m_idx == mesh_idx]

    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax_gt, ax_pred) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    gt_vals: List[float] = []
    pred_vals: List[float] = []

    normalize_pressure = bool(getattr(dataset, "normalize_pressure", CONFIG.get("normalize_pressure", False)))
    scale = float(getattr(dataset, "pressure_scale", CONFIG.get("pressure_scale", 1.0))) if normalize_pressure else 1.0

    with torch.no_grad():
        for idx in sample_indices:
            sample = dataset[idx]
            graph = sample["graph"].clone().to(DEVICE)
            dynamic = sample["dynamic"].unsqueeze(0).to(DEVICE)
            times = sample["time_values"].cpu().numpy()
            pressure_gt = sample["pressure_target"].cpu().numpy() * scale
            _, target_volume = dataset.samples[idx]

            batch_graph = Batch.from_data_list([graph])
            outputs = model(batch_graph, dynamic)
            pressure_pred = outputs["forward"]["pressure"].squeeze(0).cpu().numpy()
            if normalize_pressure:
                pressure_pred = pressure_pred * scale

            gt_vals.extend(pressure_gt.tolist())
            pred_vals.extend(pressure_pred.tolist())

            label = f"V={target_volume:.1f}"
            ax_gt.plot(times, pressure_gt, '-', alpha=0.6, linewidth=1.2, label=label)
            ax_pred.plot(times, pressure_pred, '-', alpha=0.6, linewidth=1.2, label=label)

    gt_range = f"[{min(gt_vals):.1f}, {max(gt_vals):.1f}]"
    pred_range = f"[{min(pred_vals):.1f}, {max(pred_vals):.1f}]"

    ax_gt.set_ylabel('Pressure (mmHg) - Ground Truth')
    ax_gt.set_title(f"{title_prefix} mesh{mesh_idx:02d} | GT range {gt_range}")
    ax_gt.legend(ncol=4, fontsize='x-small')
    ax_gt.grid(True, alpha=0.3)

    ax_pred.set_ylabel('Pressure (mmHg) - Prediction')
    ax_pred.set_xlabel('Time (ms)')
    ax_pred.set_title(f"Pred range {pred_range}", loc='left')
    ax_pred.legend(ncol=4, fontsize='x-small')
    ax_pred.grid(True, alpha=0.3)

    fig.tight_layout()
    filename = f"pressure_snapshot_{title_prefix.lower()}_mesh{mesh_idx:02d}.png"
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)
    model.train()


def plot_test_summary_cases(
    model: CGFENet,
    dataset: FixedSequenceDataset,
    output_dir: Path,
    *,
    title: str,
    max_cases: Optional[int] = None,
) -> None:
    """Plot all volume curves for a mesh on a single figure (one case per mesh)."""
    if len(dataset) == 0:
        return

    mesh_to_indices: Dict[int, List[int]] = {}
    for idx, (mesh_idx, _) in enumerate(dataset.samples):
        mesh_to_indices.setdefault(int(mesh_idx), []).append(idx)

    unique_meshes = sorted(mesh_to_indices.keys())
    limit = len(unique_meshes) if max_cases is None else min(len(unique_meshes), max_cases)
    model_was_training = model.training
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    for mesh_idx in unique_meshes[:limit]:
        indices = mesh_to_indices[mesh_idx]

        times_list: List[np.ndarray] = []
        volumes_list: List[np.ndarray] = []
        gt_list: List[np.ndarray] = []
        pred_list: List[np.ndarray] = []

        time_stride_plot = max(1, int(CONFIG.get("plot_time_stride", 8)))

        # Process ALL volume curves for this mesh
        for idx in indices:
            sample = dataset[idx]
            graph = sample["graph"].clone().to(DEVICE)
            dynamic = sample["dynamic"].unsqueeze(0).to(DEVICE)
            times = sample["time_values"].cpu().numpy()[::time_stride_plot]
            vols = sample.get("volume_values")
            if vols is None:
                target_volume = dataset.samples[idx][1]
                vols = np.full_like(times, target_volume, dtype=np.float32)
            else:
                vols = vols.cpu().numpy()[::time_stride_plot]
            pressure_gt = sample["pressure_target"].cpu().numpy()[::time_stride_plot]

            batch_graph = Batch.from_data_list([graph])
            disp_indices = sample.get("disp_time_indices")
            if disp_indices is not None:
                disp_indices = disp_indices.to(DEVICE)
            with torch.no_grad():
                outputs = model(batch_graph, dynamic, disp_indices=disp_indices)
                pressure_pred = outputs["forward"]["pressure"].squeeze(0).cpu().numpy()[::time_stride_plot]

            times_list.append(times)
            volumes_list.append(vols)
            gt_list.append(pressure_gt)
            pred_list.append(pressure_pred)

        # Create figure with 3D plot on left and aggregated error plot on right
        fig = plt.figure(figsize=(10.0, 4.2))
        ax3d = fig.add_subplot(121, projection="3d")

        # Plot ALL volume curves (GT and Pred) for this mesh
        for times, vols, gt_seq, pred_seq in zip(times_list, volumes_list, gt_list, pred_list):
            ax3d.plot(times, vols, gt_seq, color="tab:blue", linewidth=0.9, alpha=0.85, linestyle="-")
            ax3d.plot(times, vols, pred_seq, color="tab:orange", linewidth=0.9, alpha=0.85, linestyle="--")

        ax3d.set_xlabel("Time (ms)")
        ax3d.set_ylabel("Volume (ml)")
        ax3d.set_zlabel("Pressure (mmHg)")
        ax3d.set_title(f"{title} | mesh{mesh_idx:02d}")
        ax3d.view_init(elev=25.0, azim=-60.0)
        legend_handles = [
            plt.Line2D([0], [0], color="tab:blue", linestyle="-", linewidth=1.0, label="GT"),
            plt.Line2D([0], [0], color="tab:orange", linestyle="--", linewidth=1.0, label="Pred"),
        ]
        ax3d.legend(handles=legend_handles, loc="upper left", fontsize=8)

        # Aggregated error plot (mean and std across all volume curves)
        ref_times = times_list[0]
        gt_arr = np.stack(gt_list, axis=0)
        pred_arr = np.stack(pred_list, axis=0)
        err_arr = pred_arr - gt_arr
        mean_err = err_arr.mean(axis=0)
        std_err = err_arr.std(axis=0)

        ax_err = fig.add_subplot(122)
        ax_err.plot(
            ref_times,
            mean_err,
            label=f"Mean error (min={mean_err.min():.3f}, max={mean_err.max():.3f})",
            color="tab:red",
            linewidth=1.1,
        )
        ax_err.fill_between(
            ref_times,
            mean_err - std_err,
            mean_err + std_err,
            color="tab:red",
            alpha=0.18,
            label="±1 std",
        )
        ax_err.set_xlabel("Time (ms)")
        ax_err.set_ylabel("Pressure Error (mmHg)")
        ax_err.set_title("Aggregated Pressure Error")
        ax_err.grid(True, linewidth=0.4, linestyle=":")
        ax_err.legend(loc="upper right", fontsize=8)

        fig.tight_layout(rect=[0, 0.02, 1, 0.98])
        sanitized_title = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower()).strip("_")
        save_path = output_dir / f"{sanitized_title}_mesh{mesh_idx:02d}.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

    if model_was_training:
        model.train()


def plot_warped_mesh_overlays(
    model: CGFENet,
    dataset: FixedSequenceDataset,
    output_root: Path,
    *,
    title: str,
    max_cases: Optional[int] = None,
) -> None:
    if len(dataset) == 0 or not hasattr(dataset, "samples"):
        return
    mesh_cache = getattr(dataset, "mesh_cache", {})
    if not mesh_cache:
        return

    unique_meshes = sorted({int(mesh_idx) for mesh_idx, _ in dataset.samples})
    if not unique_meshes:
        return
    limit = len(unique_meshes) if max_cases is None else min(len(unique_meshes), max_cases)
    stats = getattr(dataset, "stats", None)
    if stats is None:
        coord_mean = np.zeros(3, dtype=np.float32)
    else:
        coord_mean = np.asarray(stats.coord_mean, dtype=np.float32)

    model_was_training = model.training
    model.eval()
    output_root.mkdir(parents=True, exist_ok=True)

    def _select_sample_index(mesh_id: int) -> Optional[int]:
        candidate_indices = [idx for idx, (mid, _) in enumerate(dataset.samples) if int(mid) == mesh_id]
        if not candidate_indices:
            return None
        base_seed = int(CONFIG.get("random_seed", 42))
        rng = np.random.default_rng(base_seed + mesh_id)
        return int(candidate_indices[int(rng.integers(len(candidate_indices)))])

    def _write_ascii_stl(path: Path, name: str, verts: np.ndarray, faces: np.ndarray) -> None:
        if faces.size == 0:
            return
        with path.open("w") as f:
            f.write(f"solid {name}\n")
            for tri in faces:
                a, b, c = verts[tri[0]], verts[tri[1]], verts[tri[2]]
                n = np.cross(b - a, c - a)
                norm = np.linalg.norm(n)
                if norm > 1e-12:
                    n = n / norm
                else:
                    n = np.array([0.0, 0.0, 0.0], dtype=float)
                f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {a[0]:.6e} {a[1]:.6e} {a[2]:.6e}\n")
                f.write(f"      vertex {b[0]:.6e} {b[1]:.6e} {b[2]:.6e}\n")
                f.write(f"      vertex {c[0]:.6e} {c[1]:.6e} {c[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write(f"endsolid {name}\n")

    def _write_legacy_vtk(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
        if faces.size == 0:
            return
        # Legacy VTK PolyData ASCII
        with path.open("w") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("mesh export\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")
            f.write(f"POINTS {verts.shape[0]} float\n")
            for v in verts:
                f.write(f"{v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            ntri = faces.shape[0]
            f.write(f"POLYGONS {ntri} {ntri*4}\n")
            for tri in faces:
                f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")

    for mesh_idx in unique_meshes[:limit]:
        case_dir = output_root / f"mesh{mesh_idx:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        statics = mesh_cache.get(mesh_idx)
        if statics is None:
            continue
        # at least one surface must exist
        if statics.epi_faces.size == 0 and statics.endo_faces.size == 0:
            continue

        sample_index = _select_sample_index(mesh_idx)
        if sample_index is None:
            continue
        sample = dataset[sample_index]
        if sample.get("disp_target") is None:
            continue

        disp_target = sample["disp_target"].to(DEVICE)

        graph = sample["graph"].clone().to(DEVICE)
        dynamic = sample["dynamic"].unsqueeze(0).to(DEVICE)
        static_labels = sample["static_labels"].to(DEVICE)
        if static_labels.dim() == 1:
            static_labels = static_labels.unsqueeze(0)
        coord_std = sample["coord_std"].to(DEVICE)
        disp_indices = sample.get("disp_time_indices")
        if disp_indices is not None:
            disp_indices = disp_indices.to(DEVICE)

        batch_graph = Batch.from_data_list([graph])
        with torch.no_grad():
            outputs = model(
                batch_graph,
                dynamic,
                disp_indices=disp_indices,
                real_disp=disp_target,
                static_labels=static_labels,
            )

        ptr = batch_graph.ptr.cpu().numpy()
        start = int(ptr[0])
        end = int(ptr[1])
        coord_std_np = coord_std.cpu().numpy().reshape(1, 1, 3)
        coord_std_vec = coord_std.cpu().numpy().reshape(1, 3)

        coords_norm = batch_graph.x[start:end, :3].cpu().numpy()
        base_coords = coords_norm * coord_std_vec + coord_mean
        gt_disp_norm = disp_target[start:end].cpu().numpy()
        gt_disp_actual = gt_disp_norm * coord_std_np
        forward_disp_pred = outputs["forward"]["disp"][start:end].detach().cpu().numpy()
        pred_disp_actual = forward_disp_pred * coord_std_np

        inv_sup = outputs.get("inverse_supervised") or {}
        inv_sup_mesh = inv_sup.get("mesh")
        if inv_sup_mesh is not None:
            pred_unloaded = inv_sup_mesh[start:end].detach().cpu().numpy() * coord_std_vec + coord_mean
        else:
            cycle1_mesh_norm = outputs["cycle1"]["mesh"][start:end].detach().cpu().numpy()
            pred_unloaded = cycle1_mesh_norm * coord_std_vec + coord_mean

        gt_unloaded = base_coords
        time_values = sample["time_values"].cpu().numpy()
        volume_series = sample.get("volume_values")
        if volume_series is not None:
            volume_series = volume_series.cpu().numpy()
        frame_count = gt_disp_actual.shape[1]
        base_seed = int(CONFIG.get("random_seed", 42))
        rng_frame = np.random.default_rng(base_seed + mesh_idx * 100003 + sample_index)
        frame_idx = int(rng_frame.integers(frame_count))
        time_value = float(time_values[frame_idx])
        volume_value = float(volume_series[frame_idx]) if volume_series is not None else float(vol_value)

        gt_loaded = base_coords + gt_disp_actual[:, frame_idx, :]
        pred_loaded = base_coords + pred_disp_actual[:, frame_idx, :]
        gt_loaded_final = gt_loaded
        pred_loaded_final = pred_loaded

        # Target volume for this case and static unloaded volume
        vol_value = float(dataset.samples[sample_index][1])
        try:
            unloaded_vol_value = float(statics.static_scalars[0])
        except Exception:
            unloaded_vol_value = float('nan')

        # Compute overlap metrics for loaded state (forward displacement)
        disp_error_loaded = pred_disp_actual[:, frame_idx, :] - gt_disp_actual[:, frame_idx, :]
        disp_error_norm_loaded = np.linalg.norm(disp_error_loaded, axis=-1)
        overlap_0p1_loaded = float(np.mean(disp_error_norm_loaded <= 0.1))
        overlap_0p2_loaded = float(np.mean(disp_error_norm_loaded <= 0.2))

        # Compute overlap metrics for unloaded state (inverse)
        unloaded_error = pred_unloaded - gt_unloaded
        unloaded_error_norm = np.linalg.norm(unloaded_error, axis=-1)
        overlap_0p1_unloaded = float(np.mean(unloaded_error_norm <= 0.1))
        overlap_0p2_unloaded = float(np.mean(unloaded_error_norm <= 0.2))

        # Export four mesh files (STL and VTK) if enabled: GT/Pred for Loaded and Unloaded (full heart surfaces)
        if bool(CONFIG.get("test_export_stl", False)):
            if statics.epi_faces.size > 0 and statics.endo_faces.size > 0:
                faces_all = np.vstack([statics.epi_faces, statics.endo_faces])
            elif statics.epi_faces.size > 0:
                faces_all = statics.epi_faces
            else:
                faces_all = statics.endo_faces
            # STL
            _write_ascii_stl(case_dir / "loaded_gt.stl", f"mesh{mesh_idx:02d}_loaded_gt", gt_loaded_final, faces_all)
            _write_ascii_stl(case_dir / "loaded_pred.stl", f"mesh{mesh_idx:02d}_loaded_pred", pred_loaded_final, faces_all)
            _write_ascii_stl(case_dir / "unloaded_gt.stl", f"mesh{mesh_idx:02d}_unloaded_gt", gt_unloaded, faces_all)
            _write_ascii_stl(case_dir / "unloaded_pred.stl", f"mesh{mesh_idx:02d}_unloaded_pred", pred_unloaded, faces_all)
            # VTK (legacy polydata) optionally
            if bool(CONFIG.get("test_export_vtk", False)):
                _write_legacy_vtk(case_dir / "loaded_gt.vtk", gt_loaded_final, faces_all)
                _write_legacy_vtk(case_dir / "loaded_pred.vtk", pred_loaded_final, faces_all)
                _write_legacy_vtk(case_dir / "unloaded_gt.vtk", gt_unloaded, faces_all)
                _write_legacy_vtk(case_dir / "unloaded_pred.vtk", pred_unloaded, faces_all)

        fig = plt.figure(figsize=(10.0, 5.5))
        ax_forward = fig.add_subplot(121, projection="3d")
        ax_inverse = fig.add_subplot(122, projection="3d")

        def _add_surface(ax: plt.Axes, verts: np.ndarray, face_array: np.ndarray, color: Tuple[float, float, float, float], edge_color: Tuple[float, float, float, float]) -> None:
            if face_array.size == 0:
                return
            poly = Poly3DCollection(verts[face_array], facecolor=color, edgecolor=edge_color, linewidths=0.2)
            ax.add_collection3d(poly)

        def _add_both_surfaces(ax: plt.Axes, verts: np.ndarray, color: Tuple[float, float, float, float]) -> None:
            # draw both epi and endo using the same color (only GT vs Pred distinction)
            edge = (0.0, 0.0, 0.0, 0.35)
            _add_surface(ax, verts, statics.epi_faces, color, edge)
            _add_surface(ax, verts, statics.endo_faces, color, edge)

        gt_color = (0.2, 0.6, 1.0, 0.45)
        pred_color = (1.0, 0.3, 0.2, 0.45)

        # set full-view limits covering both GT and Pred (overall heart view)
        def _set_full_view(ax: plt.Axes, pts_list: List[np.ndarray]) -> None:
            all_pts = np.concatenate(pts_list, axis=0)
            xmin, ymin, zmin = np.min(all_pts, axis=0)
            xmax, ymax, zmax = np.max(all_pts, axis=0)
            # pad a bit
            dx = xmax - xmin; dy = ymax - ymin; dz = zmax - zmin
            pad = 0.02 * max(dx, dy, dz)
            cx = (xmax + xmin) * 0.5; cy = (ymax + ymin) * 0.5; cz = (zmax + zmin) * 0.5
            r = 0.5 * max(dx, dy, dz) + pad
            ax.set_xlim(cx - r, cx + r)
            ax.set_ylim(cy - r, cy + r)
            ax.set_zlim(cz - r, cz + r)

        _add_both_surfaces(ax_forward, gt_loaded, gt_color)
        _add_both_surfaces(ax_forward, pred_loaded, pred_color)
        ax_forward.set_title("Loaded")
        ax_forward.set_xlabel("X")
        ax_forward.set_ylabel("Y")
        ax_forward.set_zlabel("Z")
        _set_full_view(ax_forward, [gt_loaded.reshape(-1, 3), pred_loaded.reshape(-1, 3)])

        _add_both_surfaces(ax_inverse, gt_unloaded, gt_color)
        _add_both_surfaces(ax_inverse, pred_unloaded, pred_color)
        ax_inverse.set_title("Unloaded")
        ax_inverse.set_xlabel("X")
        ax_inverse.set_ylabel("Y")
        ax_inverse.set_zlabel("Z")
        _set_full_view(ax_inverse, [gt_unloaded.reshape(-1, 3), pred_unloaded.reshape(-1, 3)])

        for ax in (ax_forward, ax_inverse):
            try:
                ax.set_box_aspect((1.0, 1.0, 1.0))
            except AttributeError:
                pass
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

        legend_handles = [
            plt.Line2D([0], [0], color=gt_color, marker="s", linestyle="", label="GT"),
            plt.Line2D([0], [0], color=pred_color, marker="s", linestyle="", label="Pred"),
        ]
        ax_forward.legend(handles=legend_handles, loc="upper right", fontsize=8)

        # Add overlap text on left side for loaded state
        overlap_text_left = f"overlap@0.1: {overlap_0p1_loaded:.3f}\noverlap@0.2: {overlap_0p2_loaded:.3f}"
        fig.text(0.02, 0.5, overlap_text_left, ha="left", va="center", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

        # Add overlap text on right side for unloaded state
        overlap_text_right = f"overlap@0.1: {overlap_0p1_unloaded:.3f}\noverlap@0.2: {overlap_0p2_unloaded:.3f}"
        fig.text(0.98, 0.5, overlap_text_right, ha="right", va="center", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
        fig.savefig(case_dir / "overlay.png", dpi=220)
        plt.close(fig)

    if model_was_training:
        model.train()


def plot_loss_history(log_path: Path, output_dir: Path) -> None:
    if not log_path.exists():
        return
    # Match lines with train loss (with or without val loss)
    pattern = re.compile(r"Epoch\s+(\d+)\s+\|\s+train\s+loss=([0-9.eE+-]+)")
    epochs: List[int] = []
    losses: List[float] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                if epochs and epoch <= epochs[-1]:
                    epochs = []
                    losses = []
                epochs.append(epoch)
                losses.append(loss)

    if not epochs:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, losses, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss ((mmHg)^2)")
    ax.set_title("Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_history.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------


def split_mesh_indices(mesh_indices: Sequence[int], train_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    shuffled = list(mesh_indices)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return shuffled[:split], shuffled[split:]


def prepare_dataloaders(
    train_dataset: FixedSequenceDataset,
    val_dataset: FixedSequenceDataset,
    config: Dict[str, float],
    *,
    test_mode: bool = False,
) -> Tuple[DataLoader, DataLoader, Dict[int, MeshStatics]]:
    loader_train = None
    if not test_mode:
        loader_train = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_sequences,
        )
    loader_val = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0 if test_mode else 4,
        pin_memory=not test_mode,
        collate_fn=collate_sequences,
    )
    mesh_cache = {**train_dataset.mesh_cache, **val_dataset.mesh_cache}
    return loader_train, loader_val, mesh_cache


def load_or_build_datasets(
    mesh_indices: Sequence[int],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Tuple[NormalizationStats, FixedSequenceDataset, FixedSequenceDataset]:
    cache_config_keys = (
        "time_stride",
        "train_ratio",
        "split_seed",
        "normalize_pressure",
        "pressure_scale",
        "disp_time_gap",
    )
    current_signature = {k: CONFIG[k] for k in cache_config_keys}
    current_signature["cache_version"] = 2
    if DATASET_CACHE_PATH.exists() and not args.rebuild_cache:
        cache = torch.load(DATASET_CACHE_PATH, map_location="cpu")
        cached_signature = cache.get("config_signature")
        train_dataset = cache.get("train_dataset")
        val_dataset = cache.get("val_dataset")
        def _is_volume_stride(ds: FixedSequenceDataset) -> bool:
            if not ds.samples:
                return True
            _, token = ds.samples[0]
            return isinstance(token, float) or isinstance(token, np.floating)

        dataset_ok = (
            train_dataset is not None
            and val_dataset is not None
            and hasattr(train_dataset, "disp_time_gap")
            and hasattr(train_dataset, "store_disp_full")
            and hasattr(val_dataset, "disp_time_gap")
            and hasattr(val_dataset, "store_disp_full")
        )
        if dataset_ok:
            gap_match = (
                getattr(train_dataset, "disp_time_gap", current_signature["disp_time_gap"])
                == current_signature["disp_time_gap"]
            ) and getattr(val_dataset, "disp_time_gap", 1) == 1
        else:
            gap_match = False

        if (
            cached_signature == current_signature
            and train_dataset is not None
            and val_dataset is not None
            and _is_volume_stride(train_dataset)
            and _is_volume_stride(val_dataset)
            and dataset_ok
            and gap_match
        ):
            stats_cached = cache["stats"]
            CONFIG["static_mean"] = stats_cached.static_mean.tolist()
            CONFIG["static_std"] = stats_cached.static_std.tolist()
            logger.info(f"Loaded dataset cache from {DATASET_CACHE_PATH}")
            return stats_cached, train_dataset, val_dataset
        logger.info("Dataset cache config mismatch; rebuilding cache.")

    stats = compute_stats(mesh_indices)
    CONFIG["static_mean"] = stats.static_mean.tolist()
    CONFIG["static_std"] = stats.static_std.tolist()
    train_meshes, val_meshes = split_mesh_indices(mesh_indices, CONFIG["train_ratio"], CONFIG["split_seed"])
    normalize_pressure = bool(CONFIG.get("normalize_pressure", False))
    pressure_scale = float(CONFIG.get("pressure_scale", 1.0))
    disp_gap = max(1, int(CONFIG.get("disp_time_gap", 1)))
    train_store_full = disp_gap <= 1
    train_dataset = FixedSequenceDataset(
        train_meshes,
        stats,
        int(CONFIG["time_stride"]),
        normalize_pressure,
        pressure_scale,
        disp_time_gap=disp_gap,
        store_disp_full=train_store_full,
    )
    val_dataset = FixedSequenceDataset(
        val_meshes,
        stats,
        int(CONFIG["time_stride"]),
        normalize_pressure,
        pressure_scale,
        disp_time_gap=1,
        store_disp_full=True,
    )
    torch.save(
        {
            "stats": stats,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "config_signature": current_signature,
        },
        DATASET_CACHE_PATH,
    )
    logger.info(f"Built dataset cache at {DATASET_CACHE_PATH}")
    return stats, train_dataset, val_dataset


def run_static_pretraining(args: argparse.Namespace) -> None:
    logger = setup_static_pretrain_logger()
    seed = int(CONFIG.get("random_seed", 42))
    set_random_seed(seed)
    logger.info(f"Random seed set to {seed}")

    encoder_type = (CONFIG.get("static_encoder_type") or "mlp").lower()
    logger.info(f"Static encoder type: {encoder_type}")

    available_meshes = [
        int(p.name.replace("mesh", ""))
        for p in DATA_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("mesh")
    ]
    available_meshes.sort()
    if not available_meshes:
        logger.error("No meshes available for static encoder pretraining.")
        return

    stats, train_dataset, val_dataset = load_or_build_datasets(available_meshes, args, logger)
    mesh_cache = {**train_dataset.mesh_cache, **val_dataset.mesh_cache}
    if not mesh_cache:
        logger.error("Mesh cache is empty; cannot pretrain static encoder.")
        return

    dataset = StaticEncoderDataset(mesh_cache, stats)
    if len(dataset) == 0:
        logger.error("Static encoder dataset is empty; aborting pretraining.")
        return

    batch_size = max(1, int(CONFIG.get("pretrain_static_batch_size", 4)))
    num_epochs = max(1, int(CONFIG.get("pretrain_static_epochs", 200)))
    patience = max(0, int(CONFIG.get("pretrain_static_patience", 20)))

    generator = torch.Generator().manual_seed(seed)
    val_size = int(max(1, round(len(dataset) * 0.2))) if len(dataset) > 1 else 0
    if val_size >= len(dataset):
        val_size = max(0, len(dataset) - 1)
    if val_size > 0:
        train_size = len(dataset) - val_size
        train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_static_batches)
    else:
        train_subset = dataset
        val_subset = None
        val_loader = None

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_static_batches)

    encoder = create_static_encoder(encoder_type).to(DEVICE)
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=CONFIG.get("pretrain_static_learning_rate", 1e-3),
        weight_decay=CONFIG.get("pretrain_static_weight_decay", 0.0),
    )

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_loss = math.inf
    epochs_since_best = 0

    for epoch in range(1, num_epochs + 1):
        encoder.train()
        train_losses: List[float] = []
        for batch in train_loader:
            coords = batch["coords"].to(DEVICE)
            batch_idx = batch["batch"].to(DEVICE)
            edge_index = batch["edge_index"].to(DEVICE)
            target = batch["target"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            pred = encoder(coords, batch_idx, edge_index)
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        val_loss = train_loss
        if val_loader is not None:
            encoder.eval()
            val_losses: List[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    coords = batch["coords"].to(DEVICE)
                    batch_idx = batch["batch"].to(DEVICE)
                    edge_index = batch["edge_index"].to(DEVICE)
                    target = batch["target"].to(DEVICE)
                    pred = encoder(coords, batch_idx, edge_index)
                    val_losses.append(F.mse_loss(pred, target).item())
            val_loss = float(np.mean(val_losses)) if val_losses else train_loss

        logger.info(f"[Static] Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {key: value.cpu() for key, value in encoder.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if patience and epochs_since_best >= patience:
            logger.info(f"[Static] Early stopping triggered at epoch {epoch} (patience={patience})")
            break

    if best_state is None:
        best_state = {key: value.cpu() for key, value in encoder.state_dict().items()}

    save_path = Path(CONFIG.get("static_encoder_path") or DEFAULT_STATIC_ENCODER_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "encoder_type": encoder_type,
            "best_val_loss": best_loss,
            "config": {
                "epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": CONFIG.get("pretrain_static_learning_rate", 1e-3),
                "weight_decay": CONFIG.get("pretrain_static_weight_decay", 0.0),
                "patience": patience,
            },
        },
        save_path,
    )
    CONFIG["static_encoder_path"] = str(save_path)
    logger.info(f"Saved static encoder weights to {save_path} | best_val_loss={best_loss:.6f}")


def run_test(args: argparse.Namespace) -> None:
    logger = setup_logger(log_path=TEST_LOGGER_PATH, name="CGFENet_Test")

    # Set random seed for reproducibility
    seed = int(CONFIG.get("random_seed", 42))
    set_random_seed(seed)
    logger.info(f"Random seed set to {seed}")

    if not MODEL_PATH.exists():
        logger.error(f"Best model not found at {MODEL_PATH}; run training first.")
        return

    available_meshes = [int(p.name.replace("mesh", "")) for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("mesh")]
    available_meshes.sort()
    stats, _, test_dataset = load_or_build_datasets(available_meshes, args, logger)
    if len(test_dataset) == 0:
        logger.error("Test dataset is empty; nothing to run.")
        return

    try:
        static_encoder, static_type, static_path, _ = prepare_static_encoder(logger)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return
    logger.info(f"Static encoder for test | type={static_type} | path={static_path}")

    sample = test_dataset[0]
    sample_graph: Data = sample["graph"]
    node_features_dim = sample_graph.num_node_features
    dynamic_dim = sample["dynamic"].shape[1]
    model = CGFENet(
        node_features_dim,
        dynamic_dim,
        apply_sigmoid=CONFIG.get("normalize_pressure", False),
        fusion_mode=CONFIG.get("fusion_mode", "layernorm"),
        static_encoder=static_encoder,
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    logger.info(f"Loaded best checkpoint from {MODEL_PATH}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(CONFIG.get("test_batch_size", CONFIG["batch_size"])),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_sequences,
    )

    if len(test_loader) == 0:
        logger.warning("Test dataloader produced zero batches; nothing to run.")
        return

    logger.info(f"Test dataset: {len(test_dataset)} samples, {len(test_loader)} batches")

    # Disable expensive summary collection (Hausdorff etc.) for faster testing
    collect_summary = False
    # Temporarily set eval_disable_inverse according to test config
    _prev_disable = CONFIG.get("eval_disable_inverse", False)
    CONFIG["eval_disable_inverse"] = bool(CONFIG.get("test_eval_disable_inverse", False))
    metrics, metrics_std, summary_lists = evaluate(model, test_loader, CONFIG, collect_summary=collect_summary, show_progress=True)
    CONFIG["eval_disable_inverse"] = _prev_disable
    logger.info(
        "Test epoch (no-grad) | "
        f"loss={metrics.get('loss', float('nan')):.6f} "
        f"(P={metrics.get('pressure_loss', 0.0):.6f}, "
        f"c2P={metrics.get('cycle2_pressure_loss', 0.0):.6f}; "
        f"D={metrics.get('disp_loss', 0.0):.6f}, c1D={metrics.get('cycle1_mesh_loss', 0.0):.6f}, "
        f"c2D={metrics.get('cycle2_mesh_loss', 0.0):.6f}, invSupD={metrics.get('inverse_supervised_mesh_loss', 0.0):.6f})"
    )

    if collect_summary:
        try:
            summary_stats = summarise_metrics(summary_lists or {})
            summary_lines = format_summary_lines(summary_stats)
            logger.info("Test summary (mean±std): " + " | ".join(summary_lines))
        except Exception as exc:
            logger.warning(f"Failed to compute test summary: {exc}")

    try:
        plot_warped_mesh_overlays(
            model,
            test_dataset,
            TEST_OUTPUT_DIR / "mesh_overlays",
            title="Test Mesh Overlay",
            max_cases=CONFIG.get("test_overlay_cases_limit", CONFIG.get("overlay_cases_limit")),
        )
    except Exception as exc:
        logger.warning(f"Failed to generate mesh overlays: {exc}")

    # Also produce summary plots for a small number of meshes (default 1)
    try:
        plot_test_summary_cases(
            model,
            test_dataset,
            TEST_OUTPUT_DIR / "summary_cases",
            title="Test Summary",
            max_cases=1,
        )
    except Exception as exc:
        logger.warning(f"Failed to generate summary plots: {exc}")

def train_one_epoch(
    model: CGFENet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    config: Dict[str, float],
    logger: logging.Logger,
) -> Dict[str, float]:
    model.train()
    stats_accum: Dict[str, List[float]] = defaultdict(list)

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)

        graph_batch: Batch = batch["graph_batch"].to(DEVICE)
        dynamic = batch["dynamic"].to(DEVICE)
        disp_indices = batch.get("disp_time_indices")
        if disp_indices is not None:
            disp_indices = disp_indices.to(DEVICE)
        disp_target_tensor = batch["disp_target"].to(DEVICE)
        static_labels_tensor = batch["static_labels"].to(DEVICE)
        outputs = model(
            graph_batch,
            dynamic,
            disp_indices=disp_indices,
            real_disp=disp_target_tensor,
            static_labels=static_labels_tensor,
        )
        loss_batch: Dict[str, torch.Tensor] = {
            "graph_batch": graph_batch,
            "pressure_target": batch["pressure_target"].to(DEVICE),
            "disp_target": disp_target_tensor,
            "static_labels": static_labels_tensor,
        }
        if "disp_target_cm" in batch:
            loss_batch["disp_target_cm"] = batch["disp_target_cm"].to(DEVICE)
        if "coord_std" in batch:
            loss_batch["coord_std"] = batch["coord_std"].to(DEVICE)
        total_loss, metrics = compute_losses(
            outputs,
            loss_batch,
            config,
        )

        if not torch.isfinite(total_loss):
            logger.warning("Encountered non-finite loss; skipping batch")
            continue

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        optimizer.step()
        scheduler.step()

        for key, value in metrics.items():
            stats_accum[key].append(value)

    return {key: float(sum(values) / max(1, len(values))) for key, values in stats_accum.items()}


def evaluate(
    model: CGFENet,
    loader: DataLoader,
    config: Dict[str, float],
    *,
    collect_summary: bool = False,
    show_progress: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float], Optional[Dict[str, List[float]]]]:
    model.eval()
    running_stats: Dict[str, RunningStats] = defaultdict(RunningStats)
    summary_accum: Optional[Dict[str, List[float]]] = defaultdict(list) if collect_summary else None
    pressure_scale = float(config.get("pressure_scale", 1.0)) if config.get("normalize_pressure", False) else 1.0
    with torch.no_grad():
        graph_limit = int(config.get("test_summary_graph_limit", 0)) if collect_summary else 0
        graphs_seen = 0
        loader_iter = tqdm(loader, desc="Evaluating", leave=False) if show_progress else loader
        for batch in loader_iter:
            graph_batch: Batch = batch["graph_batch"].to(DEVICE)
            dynamic = batch["dynamic"].to(DEVICE)
            disp_indices = batch.get("disp_time_indices")
            if disp_indices is not None:
                disp_indices = disp_indices.to(DEVICE)
            disp_target_tensor = batch["disp_target"].to(DEVICE)
            static_labels_tensor = batch["static_labels"].to(DEVICE)
            pressure_target_tensor = batch["pressure_target"].to(DEVICE)
            disp_target_cm_tensor = batch.get("disp_target_cm")
            if disp_target_cm_tensor is not None:
                disp_target_cm_tensor = disp_target_cm_tensor.to(DEVICE)
            coord_std_tensor = batch.get("coord_std")
            if coord_std_tensor is not None:
                coord_std_tensor = coord_std_tensor.to(DEVICE)
            disable_inv = bool(config.get("eval_disable_inverse", False))
            outputs = model(
                graph_batch,
                dynamic,
                disp_indices=disp_indices,
                real_disp=None if disable_inv else disp_target_tensor,
                static_labels=static_labels_tensor,
            )
            loss_batch = {
                "graph_batch": graph_batch,
                "pressure_target": pressure_target_tensor,
                "disp_target": disp_target_tensor,
                "static_labels": static_labels_tensor,
            }
            if disp_target_cm_tensor is not None:
                loss_batch["disp_target_cm"] = disp_target_cm_tensor
            if coord_std_tensor is not None:
                loss_batch["coord_std"] = coord_std_tensor
            _, metrics = compute_losses(
                outputs,
                loss_batch,
                config,
            )
            for key, value in metrics.items():
                running_stats[key].update(float(value))
            if summary_accum is not None:
                accumulate_summary_metrics(
                    summary_accum,
                    outputs,
                    graph_batch,
                    pressure_target_tensor,
                    disp_target_tensor,
                    disp_target_cm_tensor,
                    coord_std_tensor,
                    disp_indices,
                    pressure_scale,
                )
                if graph_limit > 0:
                    graphs_seen += int(graph_batch.num_graphs)
                    if graphs_seen >= graph_limit:
                        break

    averaged = {key: tracker.mean for key, tracker in running_stats.items()}
    stds = {key: tracker.std for key, tracker in running_stats.items()}
    summary_lists = (
        {key: list(values) for key, values in summary_accum.items()} if summary_accum is not None else None
    )
    return averaged, stds, summary_lists


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def run_training(args: argparse.Namespace) -> None:
    logger = setup_logger()
    
    # Set random seed for reproducibility
    seed = int(CONFIG.get("random_seed", 42))
    set_random_seed(seed)
    logger.info(f"Random seed set to {seed}")

    available_meshes = [int(p.name.replace("mesh", "")) for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("mesh")]
    available_meshes.sort()
    stats, train_dataset, val_dataset = load_or_build_datasets(available_meshes, args, logger)
    train_loader, val_loader, mesh_cache = prepare_dataloaders(train_dataset, val_dataset, CONFIG, test_mode=False)

    plot_epi_endo_overview(mesh_cache, TRAIN_OUTPUT_DIR)

    try:
        static_encoder, static_type, static_path, static_loaded = prepare_static_encoder(logger)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return
    logger.info(
        f"Static encoder configured | type={static_type} | loaded={'yes' if static_loaded else 'no'} | path={static_path}"
    )

    sample_graph: Data = next(iter(train_loader))["graph_batch"]
    node_features_dim = sample_graph.num_node_features
    dynamic_dim = train_loader.dataset[0]["dynamic"].shape[1]

    model = CGFENet(
        node_features_dim,
        dynamic_dim,
        apply_sigmoid=CONFIG.get("normalize_pressure", False),
        static_encoder=static_encoder,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, CONFIG["num_epochs"] * len(train_loader)), eta_min=CONFIG["eta_min"])

    start_epoch = 1
    best_train_metric = math.inf

    if args.resume and CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_train_metric = float(
            checkpoint.get(
                "best_train_metric",
                checkpoint.get("best_val_metric", checkpoint.get("best_val_loss", math.inf)),
            )
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(DEVICE)
        logger.info(
            f"Resuming training from epoch {start_epoch - 1} | best_{BEST_MODEL_METRIC}={best_train_metric:.4f}"
        )
    elif args.resume:
        logger.warning("Resume flag set but checkpoint not found; starting fresh.")

    if start_epoch > CONFIG["num_epochs"]:
        logger.info("Checkpoint is already at or beyond configured epochs; nothing to do.")
        return

    collect_val_summary = bool(CONFIG.get("collect_val_summary", False))
    val_every_n_epochs = 10  # Run validation every 50 epochs

    for epoch in range(start_epoch, CONFIG["num_epochs"] + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, CONFIG, logger)

        # Only run validation every N epochs or on the last epoch (unless skip_validation is True)
        run_validation = not CONFIG.get("skip_validation", False) and (epoch % val_every_n_epochs == 0 or epoch == CONFIG["num_epochs"])

        if run_validation:
            val_metrics, val_stds, val_summary_lists = evaluate(
                model,
                val_loader,
                CONFIG,
                collect_summary=collect_val_summary,
            )

            # Log with validation metrics
            logger.info(
                f"Epoch {epoch:03d} | "
                f"train loss={train_metrics['loss']:.6f} "
                f"(P={train_metrics['pressure_loss']:.6f}, "
                f"c2P={train_metrics.get('cycle2_pressure_loss', 0.0):.6f}; "
                f"D={train_metrics['disp_loss']:.6f}, c1D={train_metrics.get('cycle1_mesh_loss', 0.0):.6f}, "
                f"c2D={train_metrics.get('cycle2_mesh_loss', 0.0):.6f}, invSupD={train_metrics.get('inverse_supervised_mesh_loss', 0.0):.6f}) | "
                f"val loss={val_metrics['loss']:.6f} "
                f"(P={val_metrics['pressure_loss']:.6f}, "
                f"c2P={val_metrics.get('cycle2_pressure_loss', 0.0):.6f}; "
                f"D={val_metrics['disp_loss']:.6f}, "
                f"c1D={val_metrics.get('cycle1_mesh_loss', 0.0):.6f}, "
                f"c2D={val_metrics.get('cycle2_mesh_loss', 0.0):.6f}, "
                f"invSupD={val_metrics.get('inverse_supervised_mesh_loss', 0.0):.6f}) | "
                f"val RMSE_P(f/c2)={val_metrics['pressure_rmse']:.6f}/"
                f"{val_metrics.get('cycle2_pressure_rmse', 0.0):.6f} | "
                f"val P_r={val_metrics.get('pressure_r', 0.0):.6f} | "
                f"val P_r2={val_metrics.get('pressure_r2', 0.0):.6f} | "
                f"val RMSE_D(f/c1/c2/invSup)={val_metrics['disp_rmse']:.6f}/"
                f"{val_metrics.get('cycle1_disp_rmse', 0.0):.6f}/"
                f"{val_metrics.get('cycle2_disp_rmse', 0.0):.6f}/"
                f"{val_metrics.get('inverse_supervised_disp_rmse', 0.0):.6f} | "
                f"Overlap@0.1(f/c1/c2/invSup)={val_metrics.get('disp_overlap', 0.0):.6f}/"
                f"{val_metrics.get('cycle1_disp_overlap', 0.0):.6f}/"
                f"{val_metrics.get('cycle2_disp_overlap', 0.0):.6f}/"
                f"{val_metrics.get('inverse_supervised_disp_overlap', 0.0):.6f} | "
                f"Overlap@0.2(f/c1/c2/invSup)={val_metrics.get('disp_overlap_0p2', 0.0):.6f}/"
                f"{val_metrics.get('cycle1_disp_overlap_0p2', 0.0):.6f}/"
                f"{val_metrics.get('cycle2_disp_overlap_0p2', 0.0):.6f}/"
                f"{val_metrics.get('inverse_supervised_disp_overlap_0p2', 0.0):.6f} | "
                f"HD(f/c1/c2/invSup)={val_metrics.get('forward_hd', 0.0):.6f}/"
                f"{val_metrics.get('cycle1_hd', 0.0):.6f}/"
                f"{val_metrics.get('cycle2_hd', 0.0):.6f}/"
                f"{val_metrics.get('inverse_supervised_hd', 0.0):.6f}"
            )

            if collect_val_summary:
                summary_stats = summarise_metrics(val_summary_lists or {})
                summary_lines = format_summary_lines(summary_stats)
                logger.info("Validation summary (mean±std): " + " | ".join(summary_lines))
        else:
            # Skip validation, only log training metrics
            val_metrics = None
            val_stds = {}
            val_summary_lists = {}

            logger.info(
                f"Epoch {epoch:03d} | "
                f"train loss={train_metrics['loss']:.6f} "
                f"(P={train_metrics['pressure_loss']:.6f}, "
                f"c2P={train_metrics.get('cycle2_pressure_loss', 0.0):.6f}; "
                f"D={train_metrics['disp_loss']:.6f}, c1D={train_metrics.get('cycle1_mesh_loss', 0.0):.6f}, "
                f"c2D={train_metrics.get('cycle2_mesh_loss', 0.0):.6f}, invSupD={train_metrics.get('inverse_supervised_mesh_loss', 0.0):.6f})"
            )

        train_metric_value = float(train_metrics.get("loss", math.inf))
        if math.isfinite(train_metric_value) and train_metric_value < best_train_metric:
            best_train_metric = train_metric_value
            torch.save({"model_state": model.state_dict(), "config": CONFIG}, MODEL_PATH)
            logger.info(f"Saved checkpoint to {MODEL_PATH} ({BEST_MODEL_METRIC}={train_metric_value:.6f})")

        checkpoint_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_train_metric": best_train_metric,
            "best_val_metric": best_train_metric,
            "best_val_loss": best_train_metric,
            "config": CONFIG,
        }
        torch.save(checkpoint_payload, CHECKPOINT_PATH)

        val_mesh = CONFIG.get("snapshot_mesh", 0)
        if val_mesh is not None and val_mesh >= 0:
            plot_pressure_snapshot(
                model,
                val_dataset,
                val_mesh,
                TRAIN_OUTPUT_DIR,
                "Validation",
            )

        train_mesh = CONFIG.get("train_snapshot_mesh", 0)
        if train_mesh is not None and train_mesh >= 0:
            plot_pressure_snapshot(
                model,
                train_dataset,
                train_mesh,
                TRAIN_OUTPUT_DIR,
                "Training",
            )

        try:
            plot_loss_history(LOGGER_PATH, TRAIN_OUTPUT_DIR)
        except Exception as exc:
            logger.warning(f"Failed to update loss history plot: {exc}")

    if MODEL_PATH.exists():
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            plot_test_summary_cases(
                model,
                val_dataset,
                TRAIN_OUTPUT_DIR / "val_summary_cases",
                title="Validation/Test Summary",
                max_cases=CONFIG.get("summary_cases_limit", 16),
            )
            plot_warped_mesh_overlays(
                model,
                val_dataset,
                TRAIN_OUTPUT_DIR / "mesh_overlays" / "validation",
                title="Validation Mesh Overlay",
                max_cases=CONFIG.get("overlay_cases_limit"),
            )
        except Exception as exc:
            logger.warning(f"Failed to generate validation artifacts: {exc}")

    logger.info("Training completed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CGFENet multi-task model")
    parser.add_argument("--rebuild-cache", action="store_true", help="Placeholder for API compatibility")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint if available")
    parser.add_argument("--test", action="store_true", help="Run evaluation with the best checkpoint and exit")
    parser.add_argument("--pretrain-static", action="store_true", help="Pretrain the static encoder and exit")
    parser.add_argument(
        "--static-encoder-type",
        choices=["mlp", "graph"],
        help="Override static encoder backbone (default from config)",
    )
    parser.add_argument(
        "--static-encoder-path",
        type=str,
        help="Path to load/store static encoder weights (overrides config)",
    )
    parser.add_argument(
        "--freeze-static-head",
        action="store_true",
        help="Force freezing the static encoder during main training (load weights and keep frozen)",
    )
    parser.add_argument(
        "--train-static-head",
        action="store_true",
        help="Keep static encoder trainable during main training (disable freezing)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.static_encoder_type:
        CONFIG["static_encoder_type"] = args.static_encoder_type.lower()
    if args.static_encoder_path:
        CONFIG["static_encoder_path"] = args.static_encoder_path
    if getattr(args, "freeze_static_head", False):
        CONFIG["freeze_static_head"] = True
    if args.train_static_head:
        CONFIG["freeze_static_head"] = False
    if not CONFIG.get("static_encoder_path"):
        CONFIG["static_encoder_path"] = str(DEFAULT_STATIC_ENCODER_PATH)

    if args.pretrain_static:
        run_static_pretraining(args)
        sys.exit(0)

    want_test = bool(args.test or CONFIG.get("test_mode"))
    CONFIG["test_mode"] = want_test
    if want_test:
        run_test(args)
    else:
        run_training(args)
