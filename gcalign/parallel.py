"""Run the per-section work of a stage in several processes.

Every IF section is segmented, placed and warped independently of the others,
so the three slow stages parallelise cleanly across sections. Results come back
in the order the sections were given, and each one is computed by exactly the
same code as in a single process - `--jobs` changes how long a run takes, never
what it produces.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor


def available_cpus() -> int:
    """CPUs this process may actually use (respects SLURM / cgroup limits)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:                                      # not Linux
        return max(1, os.cpu_count() or 1)


def resolve_jobs(jobs: int | None, n_items: int) -> int:
    """`0` or None means "as many as there are CPUs", never more than there is work."""
    wanted = available_cpus() if not jobs or jobs < 1 else jobs
    return max(1, min(wanted, n_items))


def run_sections(function, arguments: list[tuple], jobs: int | None = 1) -> list:
    """Call `function(*args)` for each argument tuple; keep the input order."""
    workers = resolve_jobs(jobs, len(arguments))
    if workers == 1:
        return [function(*args) for args in arguments]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(function, *args) for args in arguments]
        return [future.result() for future in futures]
