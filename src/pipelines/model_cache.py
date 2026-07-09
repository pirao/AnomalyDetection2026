"""Versioned fitted-model cache for notebook and offline analysis workflows."""

from __future__ import annotations

import hashlib
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

# Identity axis for the CODE that produces a fitted model. Unlike the git sha,
# this changes ONLY when you deliberately alter model-fitting semantics (the
# detector/scorer/preprocessing math, the fingerprint recipe, or the cache
# schema). Bump it by hand when a re-fit would produce materially different
# weights; leave it untouched for refactors, renames, docs, or unrelated code
# changes. This is what lets "same data + same config + same code" dedup
# survive ordinary commits instead of minting a new version every time.
MODEL_CODE_VERSION = 1

__all__ = [
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_DATA_DIR",
    "DEFAULT_HYPERPARAMS_PATH",
    "MODEL_CODE_VERSION",
    "compute_config_hash",
    "data_digest",
    "fit_and_save",
    "git_revision",
    "list_versions",
    "load_fitted_models",
    "model_fingerprint",
]

# This module lives at src/pipelines/model_cache.py -> repo root is 2 up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "raw"
DEFAULT_HYPERPARAMS_PATH = (
    _REPO_ROOT
    / "src/anomaly_detection/model/grouped_residual/hyperparameters/norm_model_hyperparams.yaml"
)
DEFAULT_CACHE_ROOT = _REPO_ROOT / "cache/models"
_VERSION_DIR_PATTERN = re.compile(r"^v(?P<version>\d+)$")
_TMP_VERSION_DIR_PATTERN = re.compile(r"^v(?P<version>\d+)\.tmp$")


def _resolve(p: Path, base: Path = _REPO_ROOT) -> Path:
    return p if p.is_absolute() else base / p


def _log(message: str) -> None:
    """Emit a short status line for interactive notebook workflows."""
    print(message)


# -- Fingerprinting helpers (shared with mlflow_experiments.py) ----------------
# These answer the three independent "did it change?" questions for a model
# version: did the DATA change, did the CONFIG change, did the CODE change.


def data_digest(data_dir: Path | str) -> str:
    """MD5 fingerprint (12 hex chars) of all parquet files in ``data_dir``.

    Sorted by filename for determinism across OSes. The filename is hashed too,
    so renaming a file (same bytes, different name) also changes the digest.
    """
    data_dir = _resolve(Path(data_dir))
    h = hashlib.md5()
    for f in sorted(data_dir.glob("*.parquet")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def compute_config_hash(config: dict) -> str:
    """Order-independent MD5 (12 hex chars) of a JSON-serializable config dict.

    ``sort_keys=True`` means re-ordering YAML keys does not change the hash;
    ``default=str`` tolerates non-JSON values (e.g. Paths, enums).
    """
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.md5(payload).hexdigest()[:12]


def git_revision(repo_path: Path = _REPO_ROOT) -> dict[str, str]:
    """Return ``{"git_sha", "git_dirty"}`` for the repo containing ``repo_path``.

    ``git_sha`` is the short HEAD commit (12 chars). ``git_dirty`` is the string
    ``"True"``/``"False"`` - a dirty working tree means the recorded sha does NOT
    faithfully describe the code that produced the artifact. Falls back to
    ``"unknown"`` when not in a git repo or GitPython is unavailable.
    """
    try:
        import git

        repo = git.Repo(repo_path, search_parent_directories=True)
        return {
            "git_sha": repo.head.commit.hexsha[:12],
            "git_dirty": str(repo.is_dirty()),
        }
    except Exception:
        return {"git_sha": "unknown", "git_dirty": "unknown"}


def model_fingerprint(data_digest_value: str, config_hash: str, code_version: int) -> str:
    """Combine the three change axes (data + config + code) into one identity hash.

    Two artifacts with the same fingerprint were produced from the same data,
    the same config and the same model-code version - so a new version would be
    a duplicate. The code axis is ``MODEL_CODE_VERSION`` (a manual constant bumped
    only when the fitted-model semantics change), NOT the git sha: an unrelated
    commit must not invalidate an otherwise-identical model. The git sha and
    ``git_dirty`` are recorded separately as provenance / trust tags, never as
    part of identity.
    """
    payload = f"{data_digest_value}:{config_hash}:{code_version}".encode()
    return hashlib.md5(payload).hexdigest()[:12]


def _published_version_meta(cache_root: Path, version: int) -> dict:
    """Read a published version's ``meta.json``; empty dict if absent."""
    meta_path = cache_root / f"v{version}" / "meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)


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
    """Fit one GroupedResidualDetector per scenario. Returns {scenario_id: GroupedResidualDetector}."""
    from anomaly_detection.model.grouped_residual.detector import GroupedResidualDetector
    from offline_analysis.evaluation import df_to_timeseries

    models: dict[int, object] = {}
    for sid in scenario_ids:
        fit_df = pd.read_parquet(data_dir / f"sensor_data_fit_{sid}.parquet")
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
        model = GroupedResidualDetector(params_path=hyperparams_path, scenario_id=sid)
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
    skip_if_unchanged: bool = True,
) -> int:
    """Read model YAML, fit one GroupedResidualDetector per scenario, write versioned cache.

    Parameters
    ----------
    data_dir :
        Directory containing ``sensor_data_fit_{id}.parquet`` files.
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
    skip_if_unchanged :
        When ``True`` (default) and ``version`` is auto-assigned, skip refitting
        and return the existing latest version if its ``fingerprint`` (data +
        config + ``MODEL_CODE_VERSION``) matches the current inputs - avoids
        minting duplicate versions. The git state does not affect this: the git
        sha is recorded as a provenance tag only, never as part of identity.

    Returns
    -------
    int
        The new (or, when skipped, the existing unchanged) version number.
    """
    import yaml

    data_dir = _resolve(Path(data_dir))
    hyperparams_path = _resolve(Path(hyperparams_path))
    cache_root = _resolve(Path(cache_root))

    with open(hyperparams_path) as f:
        yaml_snapshot = yaml.safe_load(f) or {}

    # Fingerprint the three change axes up front so we can both record them and
    # detect a no-op refit.
    dd = data_digest(data_dir)
    cfg_hash = compute_config_hash(yaml_snapshot)
    git_info = git_revision()
    fingerprint = model_fingerprint(dd, cfg_hash, MODEL_CODE_VERSION)

    if version is None and skip_if_unchanged:
        _latest = max(_existing_cache_versions(cache_root, include_tmp=False), default=0)
        if _latest and _published_version_meta(cache_root, _latest).get("fingerprint") == fingerprint:
            _log(
                f"No change vs v{_latest} (fingerprint {fingerprint}); skipping "
                "refit. Pass skip_if_unchanged=False to force a new version."
            )
            return _latest

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
        "data_digest": dd,
        "config_hash": cfg_hash,
        "git_sha": git_info["git_sha"],
        "git_dirty": git_info["git_dirty"],
        "fingerprint": fingerprint,
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

    try:
        display_dir = final_dir.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        display_dir = final_dir.name
    _log(f"Saved to {display_dir}  ({len(models)} models)")
    return next_version


def load_fitted_models(
    version: int | str = "latest",
    cache_root: Path | str = DEFAULT_CACHE_ROOT,
) -> "tuple[dict[int, object], dict]":
    """Load pickled GroupedResidualDetector objects from a versioned cache directory.

    Parameters
    ----------
    version :
        Integer version or ``"latest"`` (default).
    cache_root :
        Root directory of the versioned model cache.

    Returns
    -------
    tuple[dict[int, GroupedResidualDetector], dict]
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

    Each dict contains: ``version``, ``created_at``, ``notes``, ``n_sensors``,
    ``baseline_scaler``, plus the change-detection fingerprints ``data_digest``,
    ``config_hash``, ``git_sha`` and ``fingerprint`` (empty for versions written
    before fingerprinting was added).
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
                "data_digest": meta.get("data_digest", ""),
                "config_hash": meta.get("config_hash", ""),
                "git_sha": meta.get("git_sha", ""),
                "fingerprint": meta.get("fingerprint", ""),
            }
        )
    return sorted(versions, key=lambda x: x["version"] or 0)
