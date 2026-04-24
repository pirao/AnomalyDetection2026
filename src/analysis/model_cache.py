"""Versioned fitted-model cache for notebook and offline analysis workflows."""

from __future__ import annotations

import json
import os
import pickle
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

MODEL_SCHEMA_VERSION = 1

__all__ = [
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_DATA_DIR",
    "DEFAULT_HYPERPARAMS_PATH",
    "fit_and_save",
    "list_versions",
    "load_fitted_models",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = _REPO_ROOT / "data"
DEFAULT_HYPERPARAMS_PATH = (
    _REPO_ROOT / "src/sample_processing/hyperparameters/norm_model_hyperparams.yaml"
)
DEFAULT_CACHE_ROOT = _REPO_ROOT / "cache/models"
_VERSION_DIR_PATTERN = re.compile(r"^v(?P<version>\d+)$")
_TMP_VERSION_DIR_PATTERN = re.compile(r"^v(?P<version>\d+)\.tmp$")


def _resolve(p: Path, base: Path = _REPO_ROOT) -> Path:
    return p if p.is_absolute() else base / p


def _log(message: str) -> None:
    """Emit a short status line for interactive notebook workflows."""
    print(message)


def _existing_cache_versions(
    cache_root: Path,
    *,
    include_tmp: bool = True,
) -> set[int]:
    """Return version numbers already present under the cache root."""
    versions: set[int] = set()
    if not cache_root.exists():
        return versions

    for path in cache_root.iterdir():
        if not path.is_dir():
            continue
        match = _VERSION_DIR_PATTERN.match(path.name)
        if match:
            versions.add(int(match.group("version")))
            continue
        if include_tmp:
            match = _TMP_VERSION_DIR_PATTERN.match(path.name)
            if match:
                versions.add(int(match.group("version")))
    return versions


def _latest_version_from_meta(cache_root: Path) -> int:
    meta_path = cache_root / "meta.json"
    if not meta_path.exists():
        return 0
    with open(meta_path) as f:
        global_meta = json.load(f)
    return int(global_meta.get("latest_version", 0) or 0)


def _next_cache_version(cache_root: Path) -> int:
    """Return the next free version after scanning metadata and on-disk directories."""
    highest_seen = max(
        [_latest_version_from_meta(cache_root), *_existing_cache_versions(cache_root)],
        default=0,
    )
    next_version = highest_seen + 1
    while next_version in _existing_cache_versions(cache_root):
        next_version += 1
    return next_version


def _clear_cache_dir(path: Path) -> None:
    """Remove a flat cache directory created by this module."""
    if not path.exists():
        return
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a cache directory at {path}")

    for child in list(path.iterdir()):
        if child.is_dir():
            raise IsADirectoryError(
                f"Cache directory {path} contains nested directories; "
                "refusing to replace it automatically."
            )
        child.unlink()


def _remove_empty_dir(path: Path) -> None:
    """Remove an already-empty directory with retries for transient Windows locks."""
    if not path.exists():
        return
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a cache directory at {path}")

    for attempt in range(5):
        try:
            path.rmdir()
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1)


def _resolve_version(version: int | str, cache_root: Path) -> int:
    """Resolve a requested version, with 'latest' tolerating stale metadata."""
    if version != "latest":
        return int(version)

    published_versions = _existing_cache_versions(cache_root, include_tmp=False)
    latest_published = max(published_versions, default=0)
    latest_meta = _latest_version_from_meta(cache_root)
    if latest_published:
        return latest_published

    if latest_meta and (cache_root / f"v{latest_meta}").exists():
        return latest_meta

    if latest_published == 0 and latest_meta == 0:
        raise FileNotFoundError(
            f"No model cache found at {cache_root}. Run fit_and_save() first."
        )

    raise FileNotFoundError(
        "Model cache metadata exists, but no published cache directory was found at "
        f"{cache_root}. Run fit_and_save() again."
    )


def _fit_all_scenarios(
    data_dir: Path,
    scenario_ids: Iterable[int],
    hyperparams_path: Path,
) -> "dict[int, object]":
    """Fit one AnomalyModel per scenario. Returns {scenario_id: AnomalyModel}."""
    from sample_processing.model.anomaly_model import AnomalyModel
    from analysis.api_replay import df_to_timeseries

    models: dict[int, object] = {}
    for sid in scenario_ids:
        fit_df = pd.read_parquet(data_dir / f"vibe_data_fit_{sid}.parquet")
        fit_df["sampled_at"] = pd.to_datetime(
            fit_df["sampled_at"],
            errors="coerce",
            utc=True,
        )
        fit_df = (
            fit_df.dropna(subset=["sampled_at"])
            .sort_values("sampled_at")
            .reset_index(drop=True)
        )
        model = AnomalyModel(params_path=hyperparams_path, scenario_id=sid)
        model.fit(df_to_timeseries(fit_df))
        models[int(sid)] = model
        _log(f"  fitted scenario {sid}")
    return models


def fit_and_save(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    scenario_ids: Iterable[int] = range(1, 30),
    hyperparams_path: Path | str = DEFAULT_HYPERPARAMS_PATH,
    cache_root: Path | str = DEFAULT_CACHE_ROOT,
    version: int | None = None,
    notes: str = "",
) -> int:
    """Read model YAML, fit one AnomalyModel per scenario, write versioned cache.

    Parameters
    ----------
    data_dir :
        Directory containing ``vibe_data_fit_{id}.parquet`` files.
    scenario_ids :
        Scenario IDs to fit (default 1..29).
    hyperparams_path :
        Path to ``norm_model_hyperparams.yaml``. The full YAML is snapshotted
        into ``meta.json`` for traceability.
    cache_root :
        Root directory for the versioned model cache (default ``cache/models/``).
    version :
        Explicit cache version to write. When ``None`` (default), the next free
        version is chosen automatically. When an integer is provided, that
        ``v{N}`` directory is replaced if it already exists.
    notes :
        Free-text note stored in ``meta.json`` (e.g. "chosen after scaler sweep").

    Returns
    -------
    int
        The new version number.
    """
    import yaml

    data_dir = _resolve(Path(data_dir))
    hyperparams_path = _resolve(Path(hyperparams_path))
    cache_root = _resolve(Path(cache_root))

    with open(hyperparams_path) as f:
        yaml_snapshot = yaml.safe_load(f) or {}

    meta_path = cache_root / "meta.json"
    next_version = int(version) if version is not None else _next_cache_version(cache_root)
    latest_version = _latest_version_from_meta(cache_root)
    tmp_dir = cache_root / f"v{next_version}.tmp"
    final_dir = cache_root / f"v{next_version}"

    if version is None:
        if tmp_dir.exists():
            raise FileExistsError(
                f"Temporary cache directory already exists: {tmp_dir}. "
                "Remove it or choose another version."
            )
        tmp_dir.mkdir(parents=True, exist_ok=False)
        write_dir = tmp_dir
    else:
        if tmp_dir.exists():
            _clear_cache_dir(tmp_dir)
            _remove_empty_dir(tmp_dir)
        final_dir.mkdir(parents=True, exist_ok=True)
        _clear_cache_dir(final_dir)
        write_dir = final_dir

    scenario_ids = list(scenario_ids)
    _log(f"Fitting {len(scenario_ids)} scenarios -> v{next_version} ...")
    models = _fit_all_scenarios(data_dir, scenario_ids, hyperparams_path)

    for sid, model in models.items():
        with open(write_dir / f"{sid}.pkl", "wb") as f:
            pickle.dump(model, f, protocol=5)

    defaults = yaml_snapshot.get("defaults", yaml_snapshot)
    version_meta: dict = {
        "version": next_version,
        "schema_version": MODEL_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "n_sensors": len(models),
        "baseline_scaler": str(defaults.get("baseline_scaler", "standard")),
        "yaml_snapshot": yaml_snapshot,
    }
    with open(write_dir / "meta.json", "w") as f:
        json.dump(version_meta, f, indent=2)

    if version is None:
        os.replace(str(tmp_dir), str(final_dir))

    tmp_meta = cache_root / "meta.json.tmp"
    with open(tmp_meta, "w") as f:
        json.dump({"latest_version": max(latest_version, next_version)}, f, indent=2)
    os.replace(str(tmp_meta), str(meta_path))

    _log(f"Saved to {final_dir}  ({len(models)} models)")
    return next_version


def load_fitted_models(
    version: int | str = "latest",
    cache_root: Path | str = DEFAULT_CACHE_ROOT,
) -> "tuple[dict[int, object], dict]":
    """Load pickled AnomalyModel objects from a versioned cache directory.

    Parameters
    ----------
    version :
        Integer version or ``"latest"`` (default).
    cache_root :
        Root directory of the versioned model cache.

    Returns
    -------
    tuple[dict[int, AnomalyModel], dict]
        ``(models_dict, meta_dict)`` where ``meta_dict`` contains the YAML
        snapshot and provenance info recorded at fit time.
    """
    cache_root = _resolve(Path(cache_root))

    version = _resolve_version(version, cache_root)

    version_dir = cache_root / f"v{version}"
    if not version_dir.exists():
        raise FileNotFoundError(f"Version {version} not found at {version_dir}")

    with open(version_dir / "meta.json") as f:
        meta = json.load(f)

    if meta.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise RuntimeError(
            f"Schema version mismatch: cache has schema_version="
            f"{meta.get('schema_version')!r}, code expects {MODEL_SCHEMA_VERSION}. "
            "Refit required - call fit_and_save() again."
        )

    models: dict[int, object] = {}
    for pkl_path in sorted(version_dir.glob("*.pkl")):
        sid = int(pkl_path.stem)
        with open(pkl_path, "rb") as f:
            models[sid] = pickle.load(f)

    _log(f"Loaded v{version}: {len(models)} models  ({meta.get('created_at', '?')[:19]})")
    return models, meta


def list_versions(
    cache_root: Path | str = DEFAULT_CACHE_ROOT,
) -> list[dict]:
    """Return a list of version metadata dicts, sorted by version number.

    Each dict contains: ``version``, ``created_at``, ``notes``,
    ``n_sensors``, ``baseline_scaler``.
    """
    cache_root = _resolve(Path(cache_root))
    if not cache_root.exists():
        return []

    versions = []
    for version in sorted(_existing_cache_versions(cache_root, include_tmp=False)):
        meta_path = cache_root / f"v{version}" / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        versions.append(
            {
                "version": meta.get("version"),
                "created_at": meta.get("created_at", ""),
                "notes": meta.get("notes", ""),
                "n_sensors": meta.get("n_sensors"),
                "baseline_scaler": meta.get("baseline_scaler"),
            }
        )
    return sorted(versions, key=lambda x: x["version"] or 0)
