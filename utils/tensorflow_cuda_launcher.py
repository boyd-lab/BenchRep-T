"""Run Python with the CUDA libraries bundled by ``tensorflow[and-cuda]``."""

from __future__ import annotations

import os
from pathlib import Path
import site
import sys


def _nvidia_paths(directory: str) -> list[str]:
    paths: list[str] = []
    for site_package in site.getsitepackages():
        paths.extend(
            str(path)
            for path in (Path(site_package) / "nvidia").glob(f"*/{directory}")
            if path.is_dir()
        )
    return sorted(set(paths))


def main() -> None:
    if not sys.argv[1:]:
        raise SystemExit("usage: python -m utils.tensorflow_cuda_launcher PYTHON_ARGS...")

    library_paths = _nvidia_paths("lib")
    binary_paths = _nvidia_paths("bin")
    if not library_paths:
        raise SystemExit(
            "TensorFlow CUDA libraries were not found; install tensorflow[and-cuda]"
        )

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        library_paths + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
    )
    if binary_paths:
        env["PATH"] = os.pathsep.join(binary_paths + [env.get("PATH", "")])

    os.execvpe(sys.executable, [sys.executable, *sys.argv[1:]], env)


if __name__ == "__main__":
    main()
