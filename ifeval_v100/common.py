"""Paths, pinned data, config loading, environment capture."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = ROOT / "third_party"
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))

CONFIG_PATH = ROOT / "configs" / "grid.yaml"
RESULTS = Path(os.environ.get("IFEVAL_RESULTS", ROOT / "results"))
RUNS = RESULTS / "runs"

OFFICIAL_DIR = THIRD_PARTY / "instruction_following_eval"
# Canonical item list = the version the paper's GPT-4 responses were made for
# (commit 066e1eda, 2023-11-27). It is also what HF google/IFEval and
# lm-evaluation-harness serve. The authors later edited key 2785 (June 2024);
# that newer file is kept as LATEST_DATA_PATH and the difference is reported.
DATA_PATH = OFFICIAL_DIR / "data" / "input_data_066e1eda_2023-11-27.jsonl"
LATEST_DATA_PATH = OFFICIAL_DIR / "data" / "input_data.jsonl"
GPT4_RESPONSES = OFFICIAL_DIR / "data" / "input_response_data_gpt4_20231107_145030.jsonl"

# Provenance of vendored code / data (see third_party/PROVENANCE.md)
OFFICIAL_SCORER_COMMIT = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
LMEVAL_TAG = "v0.4.13"
SHA256 = {
    DATA_PATH.name: "86ea91641ee9e6b891330f3f019e82c481abddbdea5945106df1dfe59757fbc2",
    LATEST_DATA_PATH.name: "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49",
    GPT4_RESPONSES.name: "0cff1d1469b774e80296bee20ca696894df35f3c6d6539f2c886d67c07599dcb",
}
HF_DATASET = ("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_data() -> None:
    for p in (DATA_PATH, LATEST_DATA_PATH, GPT4_RESPONSES):
        got = sha256(p)
        if got != SHA256[p.name]:
            raise RuntimeError(f"checksum mismatch for {p}: {got}")


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_items(limit: int | None = None) -> list[dict]:
    """The paper's 541 items, from the authors' repo (pinned commit, checksummed)."""
    verify_data()
    items = read_jsonl(DATA_PATH)
    assert len(items) == 541, len(items)
    return items[:limit] if limit else items


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _version(mod: str):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return None


def _cmd(args: list[str]):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def env_info() -> dict:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname_hash": hashlib.sha256(platform.node().encode()).hexdigest()[:12],
        "libraries": {m: _version(m) for m in [
            "torch", "vllm", "xformers", "transformers", "tokenizers", "accelerate",
            "huggingface_hub", "datasets", "nltk", "langdetect", "absl", "immutabledict",
            "numpy", "pandas"]},
        "official_scorer_commit": OFFICIAL_SCORER_COMMIT,
        "lm_eval_port_tag": LMEVAL_TAG,
        "data_sha256": SHA256,
        "repo_commit": _cmd(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "repo_dirty": bool(_cmd(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"])),
        "nvidia_smi": _cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                            "--format=csv,noheader"]),
    }
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
            info["torch_cuda"] = torch.version.cuda
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception:  # noqa: BLE001
        pass
    return info


def gpu_mem_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def resolve_hw_profile(cfg: dict, requested: str | None = None) -> str:
    name = requested or os.environ.get("HW_PROFILE") or "auto"
    if name == "auto":
        mem = gpu_mem_gb()
        name = "rtx_8gb" if 0 < mem < 12 else "v100"
    if name not in cfg["hardware_profiles"]:
        raise SystemExit(f"unknown HW_PROFILE {name}; choose from {list(cfg['hardware_profiles'])}")
    return name


def generation_config(cfg: dict, profile: str) -> dict:
    g = dict(cfg["generation"])
    g.update(cfg["hardware_profiles"].get(profile) or {})
    if os.environ.get("GPU_MEM_UTIL"):
        g["gpu_memory_utilization"] = float(os.environ["GPU_MEM_UTIL"])
    if os.environ.get("MAX_NUM_SEQS"):  # fewer concurrent sequences -> smaller vLLM profiling peak
        g["max_num_seqs"] = int(os.environ["MAX_NUM_SEQS"])
    return g


def default_model_set(cfg: dict) -> str:
    hw = resolve_hw_profile(cfg)
    if hw == "rtx_8gb":
        return "rtx_8gb"
    return "v100_32gb" if gpu_mem_gb() > 24 else "v100_16gb"
