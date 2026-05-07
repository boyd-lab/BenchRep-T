"""
Wall-time and peak-memory benchmark wrapper for AIRR-Bench methods.

This intentionally measures the existing command-line evals as black boxes, so
the numbers include data loading, featurization, cross-validation training, and
evaluation unless a smaller training budget is requested.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import select
import signal
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


AIRR_ROOT = Path(__file__).resolve().parents[1]

ALL_METHODS = (
    "deeprc",
    "deeptcr",
    "abmil",
    "ensemble_xgboost",
    "ensemble_regression",
    "giana",
    "malid",
    "emerson",
    "ostmeyer",
)

DEFAULT_METHOD_CONDA_ENVS = {
    "deeprc": "deeprc",
    "deeptcr": "deeptcr",
    "abmil": "airr-bench",
    "ensemble_xgboost": "airr-bench",
    "ensemble_regression": "airr-bench",
    "giana": "giana",
    "malid": "mal_id_lite",
    "emerson": "airr-bench",
    "ostmeyer": "airr-bench",
}

METHOD_CONDA_ENV_VARS = {
    "deeprc": "DEEPRC_CONDA_ENV",
    "deeptcr": "DEEPTCR_CONDA_ENV",
    "abmil": "ABMIL_CONDA_ENV",
    "ensemble_xgboost": "XGBOOST_CONDA_ENV",
    "ensemble_regression": "REGRESSION_CONDA_ENV",
    "giana": "GIANA_CONDA_ENV",
    "malid": "MALID_CONDA_ENV",
    "emerson": "EMERSON_CONDA_ENV",
    "ostmeyer": "OSTMEYER_CONDA_ENV",
}


@dataclass
class BenchmarkResult:
    method: str
    disease: str
    budget: str
    returncode: int
    elapsed_sec: float
    peak_cpu_rss_mb: float
    peak_gpu_memory_mb: float | None
    peak_gpu_memory_delta_mb: float | None
    start_time_utc: str
    end_time_utc: str
    command: list[str]
    log_path: str
    output_csv: str
    gpu_ids: list[int]
    sample_interval_sec: float
    env_info: dict


def _read_proc_rss_kb(pid: int) -> int:
    try:
        status = Path("/proc") / str(pid) / "status"
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return 0
    return 0


def _read_proc_children(pid: int) -> list[int]:
    children_file = Path("/proc") / str(pid) / "task" / str(pid) / "children"
    try:
        text = children_file.read_text().strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return []
    return [int(x) for x in text.split()] if text else []


def _process_tree(pid: int) -> list[int]:
    seen: set[int] = set()
    stack = [pid]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        if not (Path("/proc") / str(current)).exists():
            continue
        seen.add(current)
        stack.extend(_read_proc_children(current))
    return list(seen)


def _process_tree_rss_mb(pid: int) -> float:
    return sum(_read_proc_rss_kb(p) for p in _process_tree(pid)) / 1024.0


def _query_gpu_memory_mb(gpu_ids: list[int] | None = None) -> dict[int, float]:
    if shutil.which("nvidia-smi") is None:
        return {}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return {}

    wanted = set(gpu_ids) if gpu_ids else None
    values: dict[int, float] = {}
    for line in out.strip().splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) != 2:
            continue
        try:
            idx = int(cols[0])
            used = float(cols[1])
        except ValueError:
            continue
        if wanted is None or idx in wanted:
            values[idx] = used
    return values


class ResourceSampler:
    def __init__(self, pid: int, interval_sec: float, gpu_ids: list[int] | None):
        self.pid = pid
        self.interval_sec = interval_sec
        self.gpu_ids = gpu_ids
        self.peak_cpu_rss_mb = 0.0
        self.peak_gpu_memory_mb: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_sec * 2.0))

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval_sec)
        self.sample()

    def sample(self) -> None:
        self.peak_cpu_rss_mb = max(self.peak_cpu_rss_mb, _process_tree_rss_mb(self.pid))
        if self.gpu_ids is None:
            return
        gpu_values = _query_gpu_memory_mb(self.gpu_ids)
        if gpu_values:
            current_peak = max(gpu_values.values())
            if self.peak_gpu_memory_mb is None:
                self.peak_gpu_memory_mb = current_peak
            else:
                self.peak_gpu_memory_mb = max(self.peak_gpu_memory_mb, current_peak)


def _env_info() -> dict:
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "slurm": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("SLURM_")
        },
        "git": {},
    }
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(AIRR_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(AIRR_ROOT), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        )
        info["git"] = {"commit": commit, "dirty": dirty}
    except (subprocess.SubprocessError, OSError):
        pass
    info["gpus"] = _query_gpu_memory_mb()
    return info


def _method_uses_gpu(method: str, args: argparse.Namespace) -> bool:
    if method in {"deeprc", "deeptcr", "abmil"}:
        return True
    if method == "giana":
        return bool(args.giana_use_gpu)
    if method == "malid":
        return str(args.malid_device).startswith("cuda")
    if method == "ensemble_xgboost":
        return args.xgb_device in {"cuda", "gpu"}
    return False


def _base_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    metadata = Path(args.metadata_path) if args.metadata_path else AIRR_ROOT / "data/malid_clean/metadata.tsv"
    repertoire_dir = (
        Path(args.repertoire_data_dir)
        if args.repertoire_data_dir
        else AIRR_ROOT / "data/malid_clean/TCR"
    )
    output_dir = Path(args.output_dir)
    return metadata, repertoire_dir, output_dir


def _python_command_prefix(method: str) -> list[str]:
    env_var = METHOD_CONDA_ENV_VARS.get(method)
    conda_env = os.environ.get(env_var, "") if env_var else ""
    conda_env = conda_env or os.environ.get("METHOD_CONDA_ENV", "")
    conda_env = conda_env or DEFAULT_METHOD_CONDA_ENVS.get(method, "")

    if not conda_env:
        return [sys.executable]
    conda_exe = os.environ.get("CONDA_EXE", "conda")
    return [conda_exe, "run", "--no-capture-output", "-n", conda_env, "python"]


def _budget_flags(method: str, budget: str, args: argparse.Namespace) -> list[str]:
    if budget == "full":
        return []

    debug_reps = str(args.debug_repertoires)
    if method == "deeprc":
        if budget == "smoke":
            return [
                "--debug", "--debug_repertoires", debug_reps,
                "--n_updates", "100", "--evaluate_at", "50",
                "--sample_n_sequences", "1000",
            ]
        return ["--n_updates", str(args.deeprc_one_epoch_updates), "--evaluate_at", str(args.deeprc_one_epoch_updates)]
    if method == "deeptcr":
        if budget == "smoke":
            return ["--debug", "--debug_repertoires", debug_reps, "--epochs_min", "1", "--epochs_max", "1"]
        return ["--epochs_min", "1", "--epochs_max", "1"]
    if method == "abmil":
        if budget == "smoke":
            return ["--max_repertoires_per_class", debug_reps, "--epochs", "1", "--patience", "1"]
        return ["--epochs", "1", "--patience", "1"]
    if method == "giana":
        return ["--debug", "--debug_repertoires", debug_reps] if budget == "smoke" else []
    if method in {"ensemble_xgboost", "ensemble_regression"}:
        flag = "--debug_repertoires" if method == "ensemble_xgboost" else "--debug_repertoires"
        prefix = ["--debug"] if method == "ensemble_regression" else []
        return prefix + [flag, debug_reps] if budget == "smoke" else []
    return []


def _method_command(
    method: str,
    args: argparse.Namespace,
    output_csv: Path,
    metadata: Path,
    repertoire_dir: Path,
    method_results_dir: Path,
) -> list[str]:
    common = [
        "--metadata_path",
        str(metadata),
        "--repertoire_data_dir",
        str(repertoire_dir),
        "--target_disease",
        args.disease,
        "--output_csv",
        str(output_csv),
    ]

    if method == "deeprc":
        return [
            *_python_command_prefix(method),
            "-u",
            "-m",
            "evals.deeprc_2020_disease_classification",
            *common,
            "--results_dir",
            str(method_results_dir / "deeprc"),
            "--batch_size",
            str(args.deeprc_batch_size),
            *_budget_flags(method, args.budget, args),
        ]
    if method == "deeptcr":
        return [
            *_python_command_prefix(method),
            "-u",
            "-m",
            "evals.deeptcr_2021_disease_classification",
            *common,
            "--results_dir",
            str(method_results_dir / "deeptcr"),
            "--batch_size",
            str(args.deeptcr_batch_size),
            "--device",
            "0",
            *_budget_flags(method, args.budget, args),
        ]
    if method == "abmil":
        return [
            *_python_command_prefix(method),
            "-u",
            "-m",
            "evals.ensemble_abmil_disease_classification",
            *common,
            "--features",
            args.abmil_features,
            "--model_save_dir",
            str(method_results_dir / "abmil_models"),
            *_budget_flags(method, args.budget, args),
        ]
    if method == "ensemble_xgboost":
        xgb_device = "cuda" if args.xgb_device == "gpu" else args.xgb_device
        command = [
            *_python_command_prefix(method),
            "-u",
            "-m",
            "evals.ensemble_xgboost_disease_classification",
            *common,
            "--submodel",
            args.xgboost_submodel,
            "--kmer_size",
            str(args.kmer_size),
            "--n_jobs",
            str(args.n_jobs),
            "--xgb_device",
            xgb_device,
            "--model_save_dir",
            str(method_results_dir / "ensemble_xgboost_models"),
            *_budget_flags(method, args.budget, args),
        ]
        if args.max_folds is not None:
            command.extend(["--max_folds", str(args.max_folds)])
        if args.no_gaps:
            command.append("--no_gaps")
        return command
    if method == "ensemble_regression":
        command = [
            *_python_command_prefix(method),
            "-u",
            "-m",
            "evals.ensemble_regression_disease_classification",
            *common,
            "--submodel",
            args.regression_submodel,
            "--kmer_size",
            str(args.kmer_size),
            "--n_jobs",
            str(args.n_jobs),
            *_budget_flags(method, args.budget, args),
        ]
        if args.max_folds is not None:
            command.extend(["--max_folds", str(args.max_folds)])
        if args.no_gaps:
            command.append("--no_gaps")
        return command
    if method == "giana":
        command = [
            *_python_command_prefix(method),
            "-u",
            "-m",
            "evals.giana_2021_disease_classification",
            *common,
            "--results_dir",
            str(method_results_dir / "giana"),
            "--n_threads",
            str(args.n_threads),
            "--max_seqs_per_specimen",
            str(args.giana_max_seqs_per_specimen),
            "--exact",
            "--threshold_iso",
            str(args.giana_threshold_iso),
            *_budget_flags(method, args.budget, args),
        ]
        if args.giana_use_gpu:
            command.append("--use_gpu")
        return command
    if method == "malid":
        dataset_name = args.malid_dataset_name or f"{args.disease}_only"
        cache_dir = Path(args.malid_cache_dir) if args.malid_cache_dir else method_results_dir / "malid_cache" / dataset_name
        return [
            *_python_command_prefix(method),
            "-u",
            str(Path(args.malid_lite_root) / "malid_lite/training/train_ensemble.py"),
            "--data-dir",
            str(repertoire_dir),
            "--metadata-path",
            str(metadata),
            "--cache-dir",
            str(cache_dir),
            "--dataset-name",
            dataset_name,
            "--gene-locus",
            args.malid_gene_locus,
            "--classification-mode",
            "binary",
            "--diseases",
            args.disease,
            "--reference-class",
            args.malid_reference_class,
            "--model2-abstention-strategy",
            args.malid_model2_abstention_strategy,
            "--n-jobs",
            str(args.malid_n_jobs),
            "--model3-device",
            args.malid_device,
            "--model3-embedding-batch-size",
            str(args.malid_batch_size),
            "--output-dir",
            str(method_results_dir / "malid_lite_outputs"),
        ]
    if method == "emerson":
        return [
            *_python_command_prefix(method),
            "-u",
            "-m",
            "evals.emerson_2017_disease_classification",
            *common,
        ]
    if method == "ostmeyer":
        return [
            *_python_command_prefix(method),
            "-u",
            "-m",
            "evals.ostmeyer_2019_disease_classification",
            *common,
            "--n_restarts",
            str(args.ostmeyer_n_restarts),
        ]
    raise ValueError(f"Unknown method: {method}")


def _run_one(method: str, args: argparse.Namespace) -> BenchmarkResult:
    metadata, repertoire_dir, output_dir = _base_paths(args)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    method_dir = output_dir / run_id / method
    log_dir = method_dir / "logs"
    result_dir = method_dir / "method_outputs"
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    output_csv = result_dir / f"{method}_{args.disease}_{args.budget}_scores.csv"
    log_path = log_dir / f"{method}_{args.disease}_{args.budget}.log"
    command = _method_command(method, args, output_csv, metadata, repertoire_dir, result_dir)

    env = os.environ.copy()
    uses_gpu = _method_uses_gpu(method, args)
    if args.gpu is not None and uses_gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    elif uses_gpu and env.get("CUDA_VISIBLE_DEVICES") == "":
        raise RuntimeError(
            f"{method} is configured to use GPU, but CUDA_VISIBLE_DEVICES is empty. "
            "Pass --gpu or unset CUDA_VISIBLE_DEVICES."
        )
    elif not uses_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    if method == "deeptcr":
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if method == "malid":
        malid_root = str(Path(args.malid_lite_root))
        env["PYTHONPATH"] = (
            malid_root
            if not env.get("PYTHONPATH")
            else f"{malid_root}{os.pathsep}{env['PYTHONPATH']}"
        )

    gpu_ids = ([args.gpu] if args.gpu is not None else []) if uses_gpu else None
    gpu_baseline = _query_gpu_memory_mb(gpu_ids) if gpu_ids is not None else {}
    start_time = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    timed_out = False

    print(f"\n=== Benchmarking {method} ({args.disease}, budget={args.budget}) ===")
    print("Command:", " ".join(shlex.quote(x) for x in command))
    with log_path.open("w") as log:
        log.write("Command: " + " ".join(shlex.quote(x) for x in command) + "\n\n")
        proc = subprocess.Popen(
            command,
            cwd=str(AIRR_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        sampler = ResourceSampler(proc.pid, args.sample_interval_sec, gpu_ids)
        sampler.start()
        assert proc.stdout is not None
        try:
            while True:
                if args.max_runtime_sec is not None and (time.perf_counter() - start) >= args.max_runtime_sec:
                    timed_out = True
                    msg = (
                        f"\n[resource_benchmark] max_runtime_sec={args.max_runtime_sec} "
                        "reached; terminating method and saving sampled peaks.\n"
                    )
                    log.write(msg)
                    if args.echo:
                        print(msg, end="")
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=args.terminate_grace_sec)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    break

                ready, _, _ = select.select([proc.stdout], [], [], 0.2)
                if ready:
                    line = proc.stdout.readline()
                    if line:
                        log.write(line)
                        if args.echo:
                            print(line, end="")
                    elif proc.poll() is not None:
                        break
                elif proc.poll() is not None:
                    break

            for line in proc.stdout:
                log.write(line)
                if args.echo:
                    print(line, end="")
            returncode = proc.wait()
        finally:
            sampler.stop()

    end = time.perf_counter()
    end_time = datetime.now(timezone.utc).isoformat()
    baseline_peak = max(gpu_baseline.values()) if gpu_baseline else None
    gpu_delta = None
    if sampler.peak_gpu_memory_mb is not None and baseline_peak is not None:
        gpu_delta = max(0.0, sampler.peak_gpu_memory_mb - baseline_peak)

    result = BenchmarkResult(
        method=method,
        disease=args.disease,
        budget=args.budget,
        returncode=returncode,
        elapsed_sec=end - start,
        peak_cpu_rss_mb=sampler.peak_cpu_rss_mb,
        peak_gpu_memory_mb=sampler.peak_gpu_memory_mb,
        peak_gpu_memory_delta_mb=gpu_delta,
        start_time_utc=start_time,
        end_time_utc=end_time,
        command=command,
        log_path=str(log_path),
        output_csv=str(output_csv),
        gpu_ids=gpu_ids or [],
        sample_interval_sec=args.sample_interval_sec,
        env_info={
            **_env_info(),
            "timed_out": timed_out,
            "max_runtime_sec": args.max_runtime_sec,
            "terminate_grace_sec": args.terminate_grace_sec,
        },
    )
    result_json = method_dir / "resource_benchmark.json"
    result_json.write_text(json.dumps(asdict(result), indent=2))
    print(
        f"{method}: exit={returncode} time={result.elapsed_sec:.1f}s "
        f"cpu_peak={result.peak_cpu_rss_mb:.1f} MB "
        f"gpu_peak={result.peak_gpu_memory_mb if result.peak_gpu_memory_mb is not None else 'NA'} MB"
    )
    return result


def _write_summary(results: list[BenchmarkResult], output_dir: Path, run_id: str) -> None:
    summary_dir = output_dir / run_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / "resource_benchmark_summary.json"
    csv_path = summary_dir / "resource_benchmark_summary.csv"
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2))

    fields = [
        "method",
        "disease",
        "budget",
        "returncode",
        "elapsed_sec",
        "peak_cpu_rss_mb",
        "peak_gpu_memory_mb",
        "peak_gpu_memory_delta_mb",
        "log_path",
        "output_csv",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            writer.writerow({field: row[field] for field in fields})
    print(f"\nSummary written to {csv_path}")
    print(f"Detailed JSON written to {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=ALL_METHODS)
    parser.add_argument("--disease", default="Lupus")
    parser.add_argument("--budget", choices=["full", "one_epoch", "smoke"], default="full")
    parser.add_argument("--metadata_path", default=None)
    parser.add_argument("--repertoire_data_dir", default=None)
    parser.add_argument("--output_dir", default=str(AIRR_ROOT / "results/resource_benchmarks"))
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--gpu", type=int, default=None, help="Physical GPU id to expose via CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--sample_interval_sec", type=float, default=0.5)
    parser.add_argument("--max_runtime_sec", type=float, default=None,
                        help="Terminate each method after this many seconds but still write sampled peaks.")
    parser.add_argument("--terminate_grace_sec", type=float, default=30.0,
                        help="Seconds to wait after SIGTERM before SIGKILL when max_runtime_sec is reached.")
    parser.add_argument("--echo", action="store_true", help="Also stream method logs to the terminal.")
    parser.add_argument("--fail_fast", action="store_true")

    parser.add_argument("--debug_repertoires", type=int, default=10)
    parser.add_argument("--deeprc_batch_size", type=int, default=32)
    parser.add_argument("--deeprc_one_epoch_updates", type=int, default=1000)
    parser.add_argument("--deeptcr_batch_size", type=int, default=4)
    parser.add_argument("--abmil_features", default="full", choices=["full", "cdr3_only", "vj_only"])
    parser.add_argument("--xgboost_submodel", default="ensemble", choices=["ensemble", "kmer_only", "vj_only"])
    parser.add_argument("--regression_submodel", default="kmer_only", choices=["ensemble", "kmer_only", "vj_only"])
    parser.add_argument("--kmer_size", type=int, default=4)
    parser.add_argument("--no_gaps", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=10)
    parser.add_argument("--max_folds", type=int, default=None,
                        help="Limit outer folds for ensemble_xgboost/ensemble_regression "
                             "full-data resource probes, e.g. --max_folds 1.")
    parser.add_argument("--n_threads", type=int, default=10)
    parser.add_argument("--xgb_device", default="cpu", choices=["cpu", "cuda", "gpu"])
    parser.add_argument("--giana_max_seqs_per_specimen", type=int, default=10000)
    parser.add_argument("--giana_threshold_iso", type=float, default=7)
    parser.add_argument("--giana_use_gpu", action="store_true", help="Use GPU FAISS for GIANA.")
    parser.add_argument("--malid_lite_root", default=str(AIRR_ROOT.parent / "Mal-ID-Lite"))
    parser.add_argument("--malid_cache_dir", default=None,
                        help="Fresh Mal-ID cache directory. Default: run-specific method output cache.")
    parser.add_argument("--malid_dataset_name", default=None,
                        help="Mal-ID dataset name. Default: <disease>_only.")
    parser.add_argument("--malid_gene_locus", default="TCR", choices=["TCR", "BCR"])
    parser.add_argument("--malid_reference_class", default="Healthy/Background")
    parser.add_argument("--malid_model2_abstention_strategy", default="fill_models13_mean",
                        choices=["ensemble_abstain", "fill_0.5", "fill_models13_mean"])
    parser.add_argument("--malid_n_jobs", type=int, default=10)
    parser.add_argument("--malid_device", default="cuda")
    parser.add_argument("--malid_batch_size", type=int, default=64)
    parser.add_argument("--ostmeyer_n_restarts", type=int, default=200)
    args = parser.parse_args()
    if args.max_folds is not None and args.max_folds < 1:
        parser.error("--max_folds must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_id = run_id

    results: list[BenchmarkResult] = []
    for method in args.methods:
        result = _run_one(method, args)
        results.append(result)
        _write_summary(results, Path(args.output_dir), run_id)
        if args.fail_fast and result.returncode != 0:
            return result.returncode
    return 0 if all(r.returncode == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
