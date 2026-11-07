#!/usr/bin/env python3
"""Run the portable closed-state demo and emit an auditable JSON result.

The numerical path is C, using cmachine's canonical gpu.h API and the checked-in
CPU stub.  Python only compiles, launches, validates, hashes the source inputs,
and adds a small decimal-reparse check; it is not a second model implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
DEMO = Path(__file__).resolve().parent
SOURCE_PATHS = (
    Path("demos/closed_state/closed_state.c"),
    Path("src/gpu.h"),
    Path("src/machine.h"),
    Path("tests/gpu_stub.c"),
    Path("demos/closed_state/run.py"),
    Path("demos/closed_state/README.md"),
)
EXPECTED_PATHS = {
    "closed-compact",
    "escape-stale",
    "repair-plus",
    "repair-minus",
    "closed-plus",
    "closed-minus",
    "escape-plus",
    "escape-minus",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(command: list[str], *, cwd: Path, label: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"{label} failed with exit {result.returncode}\n"
            f"stdout:\n{result.stdout}\n\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def max_abs_prefix(candidate: list[list[float]], reference: list[list[float]], cols: int) -> float:
    return max(
        (abs(candidate[row][col] - reference[row][col])
         for row in range(min(len(candidate), len(reference)))
         for col in range(cols)),
        default=0.0,
    )


def max_abs_vector(candidate: list[float], reference: list[float]) -> float:
    return max((abs(a - b) for a, b in zip(candidate, reference)), default=0.0)


def path_map(report: dict) -> dict[str, dict]:
    paths = report.get("paths")
    if not isinstance(paths, list) or {p.get("name") for p in paths} != EXPECTED_PATHS:
        raise ValueError("C report does not contain the eight expected paths")
    return {path["name"]: path for path in paths}


def compare(candidate: dict, reference: dict) -> dict[str, float | bool]:
    rank = int(candidate["rank"])
    metrics = {
        "states_max_abs": max_abs_prefix(candidate["states"], reference["states"], rank),
        "norms_retained_max_abs": max_abs_prefix(candidate["norms"], reference["norms"], rank),
        "gates_max_abs": max_abs_prefix(candidate["gates"], reference["gates"], 4),
        "ups_max_abs": max_abs_prefix(candidate["ups"], reference["ups"], 4),
        "products_max_abs": max_abs_prefix(candidate["products"], reference["products"], 4),
        "writes_max_abs": max_abs_prefix(candidate["writes"], reference["writes"], rank),
        "rho_max_abs": max_abs_vector(candidate["rho"], reference["rho"]),
    }
    metrics["max_abs"] = max(float(value) for value in metrics.values())
    metrics["decimal_reparse_within_1e-7"] = bool(metrics["max_abs"] <= 1.0e-7)
    return metrics


def first_state_difference(candidate: dict, reference: dict, threshold: float = 1.0e-7) -> int:
    rank = int(candidate["rank"])
    for row, (left, right) in enumerate(zip(candidate["states"], reference["states"])):
        if any(abs(left[col] - right[col]) > threshold for col in range(rank)):
            return row
    return -1


def validate_report(report: dict) -> dict:
    paths = path_map(report)
    mechanics = report.get("mechanics")
    if not isinstance(mechanics, dict):
        raise ValueError("C report is missing mechanics")
    required_exact = (
        "closed_vs_plus_byte_exact",
        "closed_vs_minus_byte_exact",
        "repair_vs_escape_plus_byte_exact",
        "repair_vs_escape_minus_byte_exact",
    )
    if not all(mechanics.get(key) is True for key in required_exact):
        raise ValueError(f"C exactness check failed: {mechanics}")
    expected_first = (2, 2)
    if (
        mechanics.get("stale_vs_escape_plus_first_state_difference"),
        mechanics.get("stale_vs_escape_minus_first_state_difference"),
    ) != expected_first:
        raise ValueError(f"C escape control did not separate at update 2: {mechanics}")

    comparisons = {
        "closed_vs_plus": compare(paths["closed-compact"], paths["closed-plus"]),
        "closed_vs_minus": compare(paths["closed-compact"], paths["closed-minus"]),
        "repair_vs_escape_plus": compare(paths["repair-plus"], paths["escape-plus"]),
        "repair_vs_escape_minus": compare(paths["repair-minus"], paths["escape-minus"]),
    }
    stale_controls = {
        "stale_vs_escape_plus_first_state_difference": first_state_difference(
            paths["escape-stale"], paths["escape-plus"]
        ),
        "stale_vs_escape_minus_first_state_difference": first_state_difference(
            paths["escape-stale"], paths["escape-minus"]
        ),
    }
    if any(not item["decimal_reparse_within_1e-7"] for item in comparisons.values()):
        raise ValueError(f"decimal trajectory reparse exceeded tolerance: {comparisons}")
    if stale_controls != {
        "stale_vs_escape_plus_first_state_difference": 2,
        "stale_vs_escape_minus_first_state_difference": 2,
    }:
        raise ValueError(f"decimal escape control did not separate at update 2: {stale_controls}")
    return {
        "c_bit_exact": True,
        "expected_c_exactness": required_exact,
        "comparison_metrics": comparisons,
        "escape_control": stale_controls,
        "decimal_reparse_tolerance": 1.0e-7,
        "interpretation": (
            "C memcmp checks establish exactness for this CPU-stub fixture; "
            "Python decimal metrics are a readable reparse check."
        ),
    }


def source_manifest() -> dict[str, dict[str, int | str]]:
    manifest: dict[str, dict[str, int | str]] = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        manifest[relative.as_posix()] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="JSON file to write; parent directories are created",
    )
    parser.add_argument(
        "--cc",
        default=os.environ.get("CC", "cc"),
        help="C compiler (default: CC or cc)",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    compile_command = [
        args.cc,
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        "-ffp-contract=off",
        "-I",
        "src",
        "demos/closed_state/closed_state.c",
        "tests/gpu_stub.c",
        "-lm",
        "-o",
        "{temporary}/closed_state",
    ]
    with tempfile.TemporaryDirectory(prefix="closed_state_demo_") as temporary:
        executable = Path(temporary) / "closed_state"
        actual_compile = [
            value if value != "{temporary}/closed_state" else str(executable)
            for value in compile_command
        ]
        compiler_result = run_command(actual_compile, cwd=ROOT, label="compile")
        report_path = Path(temporary) / "c-report.json"
        run_result = run_command(
            [str(executable), str(report_path)], cwd=ROOT, label="closed-state execution"
        )
        report = json.loads(report_path.read_text())
        validation = validate_report(report)

    report["execution"] = {
        "schema": "tcm-closed-state-demo-execution-v1",
        "compiler_command": compile_command,
        "compiler_stderr": compiler_result.stderr,
        "program_stdout": run_result.stdout,
        "program_stderr": run_result.stderr,
        "cwd": ".",
        "temporary_binary": True,
    }
    report["source_files"] = source_manifest()
    report["validation"] = validation
    report["result_provenance"] = {
        "generated_by": "demos/closed_state/run.py",
        "fresh_clone_rerunnable": True,
        "model_weights_supplied": False,
        "native_model_outcomes": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    mechanics = report["mechanics"]
    escape = report["escape_observation"]
    print(
        "closed-state demo: "
        f"closed exact (+/-)={mechanics['closed_vs_plus_byte_exact']}/"
        f"{mechanics['closed_vs_minus_byte_exact']}; "
        f"repair exact (+/-)={mechanics['repair_vs_escape_plus_byte_exact']}/"
        f"{mechanics['repair_vs_escape_minus_byte_exact']}; "
        f"stale first output difference (+/-)="
        f"{mechanics['stale_vs_escape_plus_first_state_difference']}/"
        f"{mechanics['stale_vs_escape_minus_first_state_difference']}; "
        f"final full output L2={escape['full_final_declared_output_l2']}"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"closed-state demo failed: {error}", file=sys.stderr)
        raise SystemExit(2)
