#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build single-point SNOP Rosseland simulators for data elements."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "free_path_model_outputs" / "simulators"
CUBA_LIB = ROOT.parent / "Cuba-4.2.2" / "libcuba.a"
CUDA_HOME = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.0"))
CUDA_LIB = CUDA_HOME / "lib64"
LOW_Z_ARCHIVE = ROOT.parent / "Rosseland_opa_low_Z.zip"
LOW_Z_ARCHIVE_PREFIX = "Rosseland_opa_low_Z/equally/"
BUILD_INFO_NAME = "build_info.json"

ELEMENTS = {
    4: {"symbol": "Be", "n": 2, "aw": 9.01, "lbfac": 30.0},
    6: {"symbol": "C", "n": 2, "aw": 12.01, "lbfac": 1000.0},
    13: {"symbol": "Al", "n": 3, "aw": 26.98, "lbfac": 300.0},
    22: {"symbol": "Ti", "n": 4, "aw": 47.87, "lbfac": 1000.0},
    26: {"symbol": "Fe", "n": 4, "aw": 55.85, "lbfac": 1000.0},
    29: {"symbol": "Cu", "n": 4, "aw": 63.55, "lbfac": 1000.0},
    42: {"symbol": "Mo", "n": 5, "aw": 95.94, "lbfac": 1000.0},
    50: {"symbol": "Sn", "n": 5, "aw": 118.71, "lbfac": 1000.0},
    56: {"symbol": "Ba", "n": 6, "aw": 137.3, "lbfac": 1000.0},
    63: {"symbol": "Eu", "n": 6, "aw": 151.96, "lbfac": 1000.0},
    74: {"symbol": "W", "n": 6, "aw": 183.84, "lbfac": 1000.0},
    79: {"symbol": "Au", "n": 6, "aw": 196.97, "lbfac": 1000.0},
    82: {"symbol": "Pb", "n": 6, "aw": 207.2, "lbfac": 1000.0},
    92: {"symbol": "U", "n": 7, "aw": 238.0, "lbfac": 1000.0},
}

LOW_Z_ELEMENTS = {
    4: {
        "dfree": 1.0,
        "source_family": "rosseland_low_z_archive",
        "source_label": "Rosseland_opa_low_Z.zip 低 Z 异构单点程序",
    },
    13: {
        "dfree": 1.0,
        "source_family": "rosseland_low_z_archive",
        "source_label": "Rosseland_opa_low_Z.zip 低 Z 异构单点程序",
    },
}
DEFAULT_ELEMENT_SETTINGS = {
    "dfree": 10.0,
    "source_family": "current_directory_hetero",
    "source_label": "当前目录异构单点不透明度程序",
}

SYMBOL_TO_Z = {str(info["symbol"]).lower(): z for z, info in ELEMENTS.items()}
DATA_FILE_ELEMENT_HINTS = {
    "data.txt": 79,
    "data_au.txt": 79,
}


def fortran_double(value: float) -> str:
    text = f"{float(value):.12g}"
    if "e" in text.lower():
        return text.replace("e", "d").replace("E", "d")
    if "." not in text:
        text = f"{text}."
    return f"{text}d0"


def build_spec_for_element(z: int) -> dict:
    if z not in ELEMENTS:
        raise ValueError(f"unsupported element Z={z}")
    spec = dict(DEFAULT_ELEMENT_SETTINGS)
    spec.update(ELEMENTS[z])
    spec.update(LOW_Z_ELEMENTS.get(z, {}))
    spec["Z"] = int(z)
    if spec["source_family"] == "rosseland_low_z_archive":
        spec["source_archive"] = str(LOW_Z_ARCHIVE)
    return spec


def expected_build_info(z: int) -> dict:
    spec = build_spec_for_element(z)
    return {
        "Z": int(z),
        "symbol": spec["symbol"],
        "n": int(spec["n"]),
        "aw": float(spec["aw"]),
        "lbfac": float(spec["lbfac"]),
        "dfree": float(spec["dfree"]),
        "source_family": spec["source_family"],
        "source_archive": spec.get("source_archive", ""),
    }


def build_info_matches(build_dir: Path, expected: dict) -> bool:
    info_path = build_dir / BUILD_INFO_NAME
    if not info_path.exists():
        return False
    try:
        actual = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, float):
            try:
                if abs(float(actual_value) - expected_value) > 1e-12:
                    return False
            except (TypeError, ValueError):
                return False
        elif key == "source_archive":
            if Path(str(actual_value)).name != Path(str(expected_value)).name:
                return False
        elif actual_value != expected_value:
            return False
    return True


def is_build_current(z: int) -> bool:
    if z not in ELEMENTS:
        return False
    build_dir = OUTPUT_DIR / f"Z_{z:g}"
    exe = build_dir / "SNOP_op_Ross_single"
    if not exe.exists():
        return False
    spec = build_spec_for_element(z)
    if spec["source_family"] == "current_directory_hetero":
        return True
    return build_info_matches(build_dir, expected_build_info(z))


def simulator_source_label(z: int) -> str:
    return str(build_spec_for_element(z).get("source_label", DEFAULT_ELEMENT_SETTINGS["source_label"]))


def infer_z_from_data_file(path: Path) -> int | None:
    hinted = DATA_FILE_ELEMENT_HINTS.get(path.name.lower())
    if hinted is not None:
        return hinted
    match = re.fullmatch(r"data[_-]?([A-Za-z]{1,2})\.txt", path.name, re.IGNORECASE)
    if not match:
        return None
    return SYMBOL_TO_Z.get(match.group(1).lower())


def discover_data_elements(paths: list[Path] | None = None) -> dict[int, list[Path]]:
    candidates = paths or sorted(ROOT.glob("data*.txt"))
    found: dict[int, list[Path]] = {}
    for path in candidates:
        z = infer_z_from_data_file(path)
        if z is None or z not in ELEMENTS:
            continue
        found.setdefault(z, []).append(path)
    return found


def par_register_source(z: int, n: int, aw: float, lbfac: float) -> str:
    return f"""\
\tmodule par_register
\timplicit none
\tsave

\tinteger, parameter :: Zan={z}
\tinteger, parameter :: n={n}

\treal(8) :: rod, tep, tgama, lnu
\treal(8) :: Aw={aw:.12g}d0, Ni
\treal(8) :: NA=6.022d23
\treal(8) :: sig(n,n), Fnm(n-1,n), fnmin(n-1,n)
\treal(8), parameter :: pi=4.d0*DATAN(1.d0)
\treal(8) :: dfree
\tinteger :: i, j, itern, max_iter=100
\treal(8) :: tol

\treal(8) :: Qnout(0:Zan), Enout_pre(0:Zan)
\treal(8) :: ssbrnsp(0:Zan)
\treal(8) :: sbar_More, LTE_fac
\treal(8) :: ne_in, ne_out
\treal(8) :: Enout_out(0:Zan)
\treal(8) :: fne, fnep, norm_fnep
\treal(8) :: jac, dne

\treal(8) :: hplnu, lbfac={lbfac:.12g}d0
\treal(8) :: fffac, bffac, bbfac, esfac
\treal(8) :: emitot, abstot

\tinteger, parameter :: nrhs=1
\tinteger :: ipiv(1)
\tinteger :: info
\tinteger :: ios

\treal(8) :: ue, slo_ue
\treal(8) :: gru2

\tend module
"""


def copy_single_program(build_dir: Path, dfree: float) -> None:
    source = (ROOT / "SNOP_op_Ross_single.F").read_text(encoding="utf-8")
    source = source.replace("dfree=10.d0", f"dfree={fortran_double(dfree)}")
    (build_dir / "SNOP_op_Ross_single.F").write_text(source, encoding="utf-8")


def copy_standard_sources(build_dir: Path, dfree: float) -> None:
    for name in [
        "functions.F",
        "cubapar.f90",
        "gpu_bridge.f90",
        "gpu_integrand.cu",
    ]:
        shutil.copy2(ROOT / name, build_dir / name)
    for name in ["screen_constants.txt", "oscillator_strengths.txt"]:
        table_path = ROOT / name
        if table_path.exists():
            shutil.copy2(table_path, build_dir / name)
    copy_single_program(build_dir, dfree)


def copy_low_z_archive_sources(build_dir: Path, dfree: float) -> None:
    if not LOW_Z_ARCHIVE.exists():
        raise FileNotFoundError(f"low-Z archive not found: {LOW_Z_ARCHIVE}")
    with zipfile.ZipFile(LOW_Z_ARCHIVE) as archive:
        for name in [
            "functions.F",
            "cubapar.f90",
            "gpu_bridge.f90",
            "gpu_integrand.cu",
            "screen_constants.txt",
            "oscillator_strengths.txt",
        ]:
            member = LOW_Z_ARCHIVE_PREFIX + name
            with archive.open(member) as src, (build_dir / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
    copy_single_program(build_dir, dfree)


def copy_sources(build_dir: Path, spec: dict) -> None:
    if spec["source_family"] == "rosseland_low_z_archive":
        copy_low_z_archive_sources(build_dir, float(spec["dfree"]))
    else:
        copy_standard_sources(build_dir, float(spec["dfree"]))


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def build_one(z: int, force: bool = False) -> Path:
    if z not in ELEMENTS:
        raise ValueError(f"unsupported element Z={z}")
    info = build_spec_for_element(z)
    build_dir = OUTPUT_DIR / f"Z_{z:g}"
    exe = build_dir / "SNOP_op_Ross_single"
    expected_info = expected_build_info(z)
    needs_manifest = info["source_family"] != "current_directory_hetero"
    if exe.exists() and not force and (not needs_manifest or build_info_matches(build_dir, expected_info)):
        return exe

    if build_dir.exists():
        for child in build_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    build_dir.mkdir(parents=True, exist_ok=True)
    copy_sources(build_dir, info)
    (build_dir / "par_register.F").write_text(
        par_register_source(z, info["n"], info["aw"], info["lbfac"]),
        encoding="utf-8",
    )

    common = ["gfortran", "-O3", "-DREALSIZE=8"]
    run(common + ["-c", "functions.F"], build_dir)
    run(common + ["-c", "par_register.F"], build_dir)
    run(common + ["-c", "cubapar.f90"], build_dir)
    run(common + ["-c", "gpu_bridge.f90"], build_dir)
    run(
        [
            str(CUDA_HOME / "bin" / "nvcc"),
            "-O3",
            "-std=c++17",
            "-c",
            "gpu_integrand.cu",
            "-o",
            "gpu_integrand.o",
        ],
        build_dir,
    )
    run(
        common
        + [
            "-o",
            str(exe),
            "SNOP_op_Ross_single.F",
            "functions.F",
            "par_register.F",
            "cubapar.f90",
            "gpu_bridge.f90",
            "gpu_integrand.o",
            str(CUBA_LIB),
            "-lm",
            "-lstdc++",
            "-llapack",
            "-lopenblas",
            f"-L{CUDA_LIB}",
            "-lcudart",
        ],
        build_dir,
    )
    build_info = dict(expected_info)
    build_info["source_label"] = info["source_label"]
    (build_dir / BUILD_INFO_NAME).write_text(
        json.dumps(build_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return exe


def main() -> None:
    parser = argparse.ArgumentParser(description="Build single-point SNOP simulators.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--z", type=int, action="append", default=[])
    parser.add_argument(
        "--data-file",
        type=Path,
        action="append",
        default=[],
        help="data file used for element discovery; defaults to data*.txt in this folder",
    )
    parser.add_argument("--all-known", action="store_true", help="build every element supported by this script")
    parser.add_argument("--list", action="store_true", help="list discovered data elements without building")
    args = parser.parse_args()

    discovered = discover_data_elements([path.resolve() for path in args.data_file] if args.data_file else None)
    if args.list:
        for z, paths in sorted(discovered.items()):
            info = ELEMENTS[z]
            joined = ", ".join(str(path) for path in paths)
            print(f"Z={z} {info['symbol']}: {joined}")
        return

    if args.z:
        zs = sorted(set(args.z))
    elif args.all_known:
        zs = sorted(ELEMENTS)
    else:
        zs = sorted(discovered)
    if not zs:
        raise SystemExit("no supported data elements found; pass --z or use names like data_Be.txt")

    for z in zs:
        exe = build_one(z, force=args.force)
        info = ELEMENTS[z]
        paths = ", ".join(str(path.name) for path in discovered.get(z, [])) or "manual"
        print(f"Z={z} {info['symbol']} ({paths}): {exe}")


if __name__ == "__main__":
    main()
