#!/usr/bin/env python3
"""The GPU milliseconds of each prefill block (HERBERT_METAL_PFTIME=1): the autotune's rounds, then each run of the
bench, one line each with the total. usage: bench/pftime_table.py LOG"""
import re, sys
rows, order = {}, []
for l in open(sys.argv[1], errors="replace"):
    m = re.match(r"\[pftime\] autotune (\d+) rows, round (\d+), block (\d+): ([0-9.]+) ms", l)
    if m:
        k = f"autotune {int(m[1]):>4} r{m[2]}"
        if k not in order:
            order.append(k); rows[k] = []
        rows[k].append(float(m[4])); continue
    m = re.match(r"\[pftime\] block (\d+)/(\d+) rows (\d+) at (\d+): ([0-9.]+) ms", l)
    if m:
        if m[1] == "0":
            k = f"run {sum(o.startswith('run') for o in order) + 1}"; order.append(k); rows[k] = []
        rows[order[-1]].append(float(m[5])); continue
    m = re.match(r"\[pp\d+\] (warm-up|run) [0-9.]+ s, ([0-9.]+) tok/s", l)
    if m and order and order[-1].startswith("run"):
        order[-1] += f" ({m[1]}, {float(m[2]):.0f} tok/s)"; rows[order[-1]] = rows.pop(order[-1].split(" (")[0])
for k in order:
    v = rows[k]
    print(f"{k:34} total {sum(v):7.0f} ms | " + " ".join(f"{x:5.0f}" for x in v))
