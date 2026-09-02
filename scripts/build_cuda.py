"""Configure and build strainX's native C++/CUDA runtime.

Usage:
    python scripts/build_cuda.py
    python scripts/build_cuda.py --arch 120

The default ``native`` architecture produces the smallest, fastest local build.
Release packagers can pass a semicolon-separated architecture list.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "strainx" / "core" / "native_cuda"
BUILD = ROOT / "build" / "native_cuda"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch", default="native",
        help="CMake CUDA architecture or list, e.g. native, 120, or 86;89;120")
    parser.add_argument("--config", default="Release", choices=("Release", "Debug"))
    parser.add_argument("--clean", action="store_true", help="Remove only the native build directory first")
    args = parser.parse_args()

    if shutil.which("cmake") is None:
        raise SystemExit("CMake was not found on PATH.")
    if shutil.which("nvcc") is None:
        raise SystemExit("nvcc was not found. Install the NVIDIA CUDA Toolkit and reopen the terminal.")
    if args.clean and BUILD.exists():
        resolved = BUILD.resolve()
        if resolved != (ROOT / "build" / "native_cuda").resolve():
            raise SystemExit(f"Refusing to remove unexpected path: {resolved}")
        shutil.rmtree(resolved)

    configure = [
        "cmake", "-S", str(SOURCE), "-B", str(BUILD),
        f"-DCMAKE_CUDA_ARCHITECTURES={args.arch}",
    ]
    if os.name == "nt":
        configure += ["-A", "x64"]
    else:
        configure += [f"-DCMAKE_BUILD_TYPE={args.config}"]
    run(configure)
    run(["cmake", "--build", str(BUILD), "--config", args.config, "--parallel"])

    names = ("strainx_cuda.dll", "libstrainx_cuda.so", "libstrainx_cuda.dylib")
    products = [path for name in names for path in (BUILD / "bin").glob(f"**/{name}")]
    if not products:
        raise SystemExit(
            "Build completed but the native CUDA library was not found under "
            f"{BUILD / 'bin'}.")
    print(f"Built {products[0]}")
    print("Restart strainX; the Parameters page will detect the native backend.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
