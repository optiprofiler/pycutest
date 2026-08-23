"""Collect or verify reviewed default SIF parameters for variable-size problems."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "sif_defaults_pycutest.json"
RUNTIME_VERSIONS = ROOT / ".github/actions/collect_info/runtime-versions.env"


def _runtime_matrix():
    values = {}
    for line in RUNTIME_VERSIONS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value
    components = {
        "pycutest": "PYCUTEST",
        "archdefs": "ARCHDEFS",
        "sifdecode": "SIFDECODE",
        "cutest": "CUTEST",
        "mastsif": "MASTSIF",
    }
    return {
        component: {
            "version": values[f"{prefix}_VERSION"],
            "commit": values[f"{prefix}_REF"],
        }
        for component, prefix in components.items()
    }


def collect():
    sys.path.insert(0, str(ROOT))
    from pycutest_tools import pycutest_get_sif_params

    with (ROOT / "probinfo_pycutest.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    variable_names = sorted(row["problem_name"] for row in rows if row["argins"])

    problems = {}
    for problem_name in variable_names:
        names, _, defaults = pycutest_get_sif_params(problem_name)
        if not names or any(default is None for default in defaults):
            raise RuntimeError(
                f"Missing reviewed SIF defaults for variable-size problem {problem_name}."
            )
        problems[problem_name] = dict(zip(names, defaults))

    return {
        "schema_version": 1,
        "runtime_matrix": _runtime_matrix(),
        "problems": problems,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    collected = collect()
    if args.check:
        committed = json.loads(args.output.read_text(encoding="utf-8"))
        if committed != collected:
            raise SystemExit(
                "Committed SIF-default metadata differs from the reviewed runtime matrix."
            )
        print(f"Verified {len(collected['problems'])} variable-size SIF defaults.")
        return

    args.output.write_text(
        json.dumps(collected, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(collected['problems'])} defaults to {args.output}.")


if __name__ == "__main__":
    main()
