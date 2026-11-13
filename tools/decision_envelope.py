"""Exact upper envelopes of supplied affine readout coefficients.

No model imports. Coefficients may be ints, finite JSON floats, decimal strings
or rational strings. A float denotes its exact stored binary64 value. Native
rounding and the provenance/completeness of supplied rows remain separate.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path


def rational(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("coefficient must be an integer, finite float or rational string")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite coefficient")
    if isinstance(value, str) and len(value) > 8192:
        raise ValueError("coefficient text exceeds inspection bound")
    result = Fraction(value)
    if result.numerator.bit_length() > 4096 or result.denominator.bit_length() > 4096:
        raise ValueError("coefficient precision exceeds inspection bound")
    return result


def _probe(left, right):
    midpoint = (left + right) / 2
    # Choose an interior dyadic point; never silently label a rounded boundary
    # as an interior native probe. Extremely narrow regions remain reportable.
    for power in range(1075):
        scale = 1 << power
        integer = (left * scale).__floor__() + 1
        point = Fraction(integer, scale)
        if point >= right:
            continue
        try:
            value = float(point)
        except OverflowError:
            continue
        if math.isfinite(value) and left < Fraction(value) < right:
            return {"exact_midpoint": str(midpoint), "binary64_dose": value,
                    "binary64_exact": str(Fraction(value))}
    return {"exact_midpoint": str(midpoint), "binary64_dose": None,
            "reason": "no interior binary64 probe found"}


def envelope(rows, lower=0, upper=1, *, vocabulary_size=None):
    lower, upper = rational(lower), rational(upper)
    if lower > upper or not rows or len(rows) > 1_000_000:
        raise ValueError("invalid envelope extent")
    lines = []
    ids = set()
    for row in rows:
        token = row["token_id"]
        if isinstance(token, bool) or not isinstance(token, int) or token < 0 or token in ids:
            raise ValueError("token IDs must be unique nonnegative integers")
        ids.add(token)
        lines.append({"a": rational(row["intercept"]), "b": rational(row["slope"]), "ids": [token]})
    if vocabulary_size is not None:
        if isinstance(vocabulary_size, bool) or not isinstance(vocabulary_size, int) or vocabulary_size < 1:
            raise ValueError("invalid vocabulary size")
        if ids != set(range(vocabulary_size)):
            raise ValueError("full-vocabulary declaration requires every vocabulary row")
    # Dominated parallels cannot win. Identical lines keep all tied IDs.
    parallel = {}
    for line in lines:
        old = parallel.get(line["b"])
        if old is None or line["a"] > old["a"]:
            parallel[line["b"]] = {"a": line["a"], "b": line["b"], "ids": list(line["ids"])}
        elif line["a"] == old["a"]:
            old["ids"].extend(line["ids"])
    hull = []
    touches = {}
    for line in sorted(parallel.values(), key=lambda x: x["b"]):
        line["ids"].sort()
        start = None
        while hull:
            old, previous_start = hull[-1]
            start = (old["a"] - line["a"]) / (line["b"] - old["b"])
            if previous_start is None or start > previous_start:
                break
            if start == previous_start:
                touches.setdefault(start, set()).update(old["ids"])
            hull.pop()
        hull.append((line, start if hull else None))
    segments = []
    for i, (line, start) in enumerate(hull):
        stop = hull[i+1][1] if i+1 < len(hull) else None
        left = lower if start is None else max(lower, start)
        right = upper if stop is None else min(upper, stop)
        if left < right:
            segments.append({"left": str(left), "right": str(right),
                             "winner": min(line["ids"]), "tied_line_ids": line["ids"],
                             "intercept": str(line["a"]), "slope": str(line["b"]),
                             "probe": _probe(left, right)})
    points = {}
    # Endpoint ties include lines whose open winning region is outside domain.
    for point in {lower, upper}:
        scores = [(line["a"] + point*line["b"], line["ids"][0]) for line in lines]
        best = max(score for score, _ in scores)
        points[point] = sorted(token for score, token in scores if score == best)
    for i in range(1, len(hull)):
        point = hull[i][1]
        if lower < point < upper:
            tied = set(hull[i-1][0]["ids"]) | set(hull[i][0]["ids"]) | touches.get(point, set())
            points[point] = sorted(tied)
    return {"schema": "tcm-decision-envelope-v1", "arithmetic": "exact-rational-supplied-coefficients",
            "coverage": "all-supplied-vocabulary-rows" if vocabulary_size is not None else "provided-lines-only",
            "vocabulary_size": vocabulary_size, "input_rows": len(rows),
            "domain": [str(lower), str(upper)], "segments": segments,
            "points": [{"dose": str(point), "tied_token_ids": tied, "winner": min(tied)}
                       for point, tied in sorted(points.items())],
            "operator_contract": "fixed-additive-final-readout/common-positive-RMS/bias-free-head",
            "native_measurement": False,
            "limitations": ["Coefficient provenance and operator contract are caller declarations.",
                            "Native arithmetic may shift decisions near boundaries.",
                            "Nonlinear suffixes and sampled decoding do not obey this affine ranking law."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON {lines:[{token_id,intercept,slope}], vocabulary_size?:int}")
    parser.add_argument("--lower", default="0")
    parser.add_argument("--upper", default="1")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    report = envelope(data["lines"], args.lower, args.upper, vocabulary_size=data.get("vocabulary_size"))
    if "source" in data:
        report["source"] = data["source"]
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
