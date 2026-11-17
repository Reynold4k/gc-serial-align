#!/usr/bin/env python
"""Compare two run directories, number for number.

    python tests/regression/compare_runs.py REFERENCE_RUN NEW_RUN

Used to show that a change to the pipeline did not change its results: every
array, table and report the reference run holds must be found unchanged in the
new run. Files and keys that only the new run has are listed, not judged - a
newer version is allowed to write more, never to write something different.

Exit status 0 when nothing the reference holds has changed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# files renamed between versions: reference name -> new name
RENAMED = {"03_gc_match/gc_presence_9plane.csv": "03_gc_match/gc_presence.csv"}
# not results: they hold a timestamp, a version, or checksums of the above
SKIP = {"SHA256SUMS.txt"}
# written by tools outside the pipeline (optional viewer layers, ad-hoc QC plots)
SKIP_PREFIXES = ("qc_", "zone_block")
SKIP_DIRS = {"05_viewer", "06_figures", "06_transfer", "06_registered_images", "07_film"}
# report keys that are bookkeeping, not results
VOLATILE_KEYS = {"generated", "gcalign_version", "frozen_inputs"}
# report keys that hold explanatory text; a change of wording is listed, not counted
TEXT_KEYS = {"note", "caveats"}
# report keys renamed between versions: reference name -> new name
RENAMED_KEYS = {"n_on_all_9_planes": "n_on_all_planes",
                "n_on_ge_5_planes": "n_on_majority_of_planes"}
TEXT_CHANGES: list[str] = []


def same_value(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (a != a and b != b)
    return a == b


def json_differences(a, b, path="") -> list[str]:
    """Paths at which `b` does not hold what `a` holds."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key, value in a.items():
            if key in VOLATILE_KEYS:
                continue
            if key in TEXT_KEYS:
                if b.get(key) != value:
                    TEXT_CHANGES.append(f"{path}/{key}")
                continue
            other = key if key in b else RENAMED_KEYS.get(key, key)
            if other not in b:
                out.append(f"{path}/{key}: missing")
            else:
                out.extend(json_differences(value, b[other], f"{path}/{key}"))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} -> {len(b)}")
        else:
            for index, (x, y) in enumerate(zip(a, b)):
                out.extend(json_differences(x, y, f"{path}[{index}]"))
    elif not same_value(a, b):
        out.append(f"{path}: {a!r} -> {b!r}")
    return out


def compare_npz(ref: Path, new: Path) -> tuple[bool, str]:
    a, b = np.load(ref, allow_pickle=False), np.load(new, allow_pickle=False)
    missing = sorted(set(a.files) - set(b.files))
    changed = [k for k in a.files if k in b.files
               and not (a[k].dtype == b[k].dtype and a[k].shape == b[k].shape
                        and np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"))]
    extra = sorted(set(b.files) - set(a.files))
    note = f"{len(a.files)} arrays"
    if extra:
        note += f", +{len(extra)} new"
    if missing or changed:
        return False, f"{note}; missing {missing[:4]}, changed {changed[:4]}"
    return True, note


def compare_csv(ref: Path, new: Path) -> tuple[bool, str]:
    import pandas as pd
    a, b = pd.read_csv(ref), pd.read_csv(new)
    missing = [c for c in a.columns if c not in b.columns]
    if missing or len(a) != len(b):
        return False, f"rows {len(a)} -> {len(b)}, missing columns {missing}"
    changed = [c for c in a.columns if not a[c].equals(b[c])]
    extra = [c for c in b.columns if c not in a.columns]
    note = f"{len(a)} rows x {len(a.columns)} columns" + (f", +{len(extra)} new" if extra else "")
    return (not changed), note + (f"; changed {changed}" if changed else "")


def compare_json(ref: Path, new: Path) -> tuple[bool, str]:
    a, b = json.loads(ref.read_text()), json.loads(new.read_text())
    if isinstance(a, list) and isinstance(b, list) and a and isinstance(a[0], dict):
        differences = []
        if len(a) != len(b):
            differences.append(f"length {len(a)} -> {len(b)}")
        for index, (x, y) in enumerate(zip(a, b)):
            differences.extend(json_differences(x, y, f"[{index}]"))
    else:
        differences = json_differences(a, b)
    return (not differences), ("; ".join(differences[:6]) + (" ..." if len(differences) > 6 else "")
                               if differences else "every reference value unchanged")


def main(reference: str, new: str) -> int:
    ref_root, new_root = Path(reference), Path(new)
    failures = 0
    files = sorted(p for p in ref_root.rglob("*") if p.is_file())
    for path in files:
        rel = path.relative_to(ref_root).as_posix()
        if rel in SKIP or rel.split("/")[0] in SKIP_DIRS or path.suffix in {".h5ad", ".bak"} \
                or ".bak_" in path.name or path.name.startswith(SKIP_PREFIXES):
            continue
        other = new_root / RENAMED.get(rel, rel)
        if not other.exists():
            other = new_root / rel                      # the new run may be an old version too
        if not other.exists():
            print(f"  MISSING    {rel}")
            failures += 1
            continue
        if path.suffix == ".npz":
            ok, note = compare_npz(path, other)
        elif path.suffix == ".csv":
            ok, note = compare_csv(path, other)
        elif path.suffix == ".json":
            ok, note = compare_json(path, other)
        else:
            ok = path.read_bytes() == other.read_bytes()
            note = "bytes"
        if ok and path.suffix in {".csv", ".json"} and path.read_bytes() == other.read_bytes():
            note += " (byte-identical)"
        print(f"  {'same' if ok else 'DIFFERENT':<10} {rel}  [{note}]")
        failures += 0 if ok else 1
    known = {RENAMED.get(p.relative_to(ref_root).as_posix(), p.relative_to(ref_root).as_posix())
             for p in files}
    added = sorted(p.relative_to(new_root).as_posix() for p in new_root.rglob("*")
                   if p.is_file() and p.relative_to(new_root).as_posix() not in known
                   and p.relative_to(new_root).as_posix().split("/")[0] not in SKIP_DIRS
                   and p.relative_to(new_root).as_posix() not in RENAMED
                   and p.name not in SKIP and not p.name.startswith(SKIP_PREFIXES))
    if added:
        print("  new files  " + ", ".join(added))
    if TEXT_CHANGES:
        print(f"  reworded   {len(TEXT_CHANGES)} explanatory text(s): " + ", ".join(TEXT_CHANGES))
    print(f"\n{'UNCHANGED' if not failures else 'CHANGED'}: "
          f"{failures} of the reference's files hold a different result")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
