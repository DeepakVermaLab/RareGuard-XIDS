from __future__ import annotations

import importlib
import os
import shutil
import sys
import time
from pathlib import Path


FEDERATED_ARTIFACTS = (
    Path("artifacts/results/federated_summary.csv"),
    Path("artifacts/results/federated_round_metrics.csv"),
    Path("artifacts/logs/run_federated.log"),
)


def _copy_file(source: Path, destination: Path, attempts: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            shutil.copy2(source, destination)
            if destination.stat().st_size != source.stat().st_size:
                raise OSError(f"Incomplete copy: {source} -> {destination}")
            return
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(float(attempt))


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Required directory not found: {source}")
    shutil.copytree(source, destination, dirs_exist_ok=True)


def stage_runtime(drive_root: Path, runtime_root: Path) -> Path:
    drive_root = drive_root.resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    _copy_tree(drive_root / "src", runtime_root / "src")
    _copy_tree(drive_root / "configs", runtime_root / "configs")

    source_processed = drive_root / "artifacts" / "processed"
    runtime_processed = runtime_root / "artifacts" / "processed"
    for source in sorted(source_processed.glob("*/features.npz")):
        destination = runtime_processed / source.parent.name / source.name
        if destination.exists() and destination.stat().st_size == source.stat().st_size:
            continue
        print(f"Staging {source.parent.name}/{source.name} to Colab local disk")
        _copy_file(source, destination)

    for name in ("logs", "results", "models", "features", "figures", "reports"):
        (runtime_root / "artifacts" / name).mkdir(parents=True, exist_ok=True)

    for relative_path in FEDERATED_ARTIFACTS:
        source = drive_root / relative_path
        destination = runtime_root / relative_path
        if source.exists() and not destination.exists():
            _copy_file(source, destination)

    return runtime_root


def _drive_project_available(drive_root: Path) -> bool:
    try:
        return drive_root.is_dir() and (drive_root / "pyproject.toml").is_file()
    except OSError:
        return False


def remount_drive(drive_root: Path) -> None:
    if _drive_project_available(drive_root):
        return
    try:
        from google.colab import drive
    except ImportError as exc:
        raise OSError(f"Google Drive project is unavailable: {drive_root}") from exc

    os.chdir("/content")
    try:
        drive.flush_and_unmount()
    except Exception:
        pass
    drive.mount("/content/drive", force_remount=True)
    if not _drive_project_available(drive_root):
        raise OSError(f"Google Drive project is still unavailable after remount: {drive_root}")


def sync_federated_artifacts(runtime_root: Path, drive_root: Path, remount: bool = False) -> None:
    if runtime_root.resolve() == drive_root.resolve():
        print("Federated checkpoints already reside in the canonical project")
        return
    if remount:
        remount_drive(drive_root)
    for relative_path in FEDERATED_ARTIFACTS:
        source = runtime_root / relative_path
        if source.exists():
            _copy_file(source, drive_root / relative_path)
    print("Federated checkpoints synced to Google Drive")


def _load_local_federated_main(runtime_root: Path):
    local_src = str(runtime_root / "src")
    sys.path.insert(0, local_src)
    os.environ["PYTHONPATH"] = local_src + os.pathsep + os.environ.get("PYTHONPATH", "")
    for module_name in list(sys.modules):
        if module_name == "rareguard" or module_name.startswith("rareguard."):
            del sys.modules[module_name]
    importlib.invalidate_caches()
    from rareguard.experiments.run_federated import main

    return main


def run_resilient_federated(
    drive_root: str | Path,
    dataset: str = "all",
    device: str = "cpu",
    runtime_root: str | Path | None = None,
) -> Path:
    drive_root = Path(drive_root).resolve()
    in_colab = Path("/content").is_dir()
    if in_colab:
        runtime_root = Path(runtime_root or "/content/rareguard_xids_runtime")
        runtime_root = stage_runtime(drive_root, runtime_root)
    else:
        if runtime_root is not None and Path(runtime_root).resolve() != drive_root:
            raise ValueError("A separate staged runtime is supported only in Colab.")
        runtime_root = drive_root
    run_federated = _load_local_federated_main(runtime_root)
    previous_cwd = Path.cwd()
    os.chdir(runtime_root)
    run_error: BaseException | None = None
    try:
        run_federated(
            [
                "--config",
                "configs/default.yaml",
                "--datasets",
                "configs/datasets.yaml",
                "--dataset",
                dataset,
                "--device",
                device,
                "--resume",
            ]
        )
    except BaseException as exc:
        run_error = exc
        raise
    finally:
        os.chdir("/content" if in_colab else previous_cwd)
        try:
            sync_federated_artifacts(runtime_root, drive_root, remount=True)
        except Exception as sync_error:
            print(f"Drive sync failed; local checkpoints remain at {runtime_root}: {sync_error}")
            if run_error is None:
                raise
    return runtime_root


if __name__ == "__main__":
    raise SystemExit("Run this helper from notebook 05 with run_resilient_federated().")
