#!/usr/bin/env python3
"""What a Metal System Trace says of a prefill: the Compute intervals of the
GPU (one per command buffer or encoder: count, each one's duration, the gaps
between consecutive ones), and the GPU's performance states (Maximum, Medium,
Minimum: the share of the busy window spent in each, and per interval the
share at Maximum). Uses bench/xctrace_table.py's export.

usage: bench/xctrace_gpu.py TRACE [--min-ms 1] [--list]
"""
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from xctrace_table import export  # noqa: E402


def rows(trace, schema):
    root = export(trace, schema, "1")
    node = root.find("node")
    if node is None:
        return [], []
    cols = [c.findtext("mnemonic") for c in node.find("schema").findall("col")]
    byid = {}

    def val(e):
        if "ref" in e.attrib:
            e = byid[e.attrib["ref"]]
        if "id" in e.attrib:
            byid[e.attrib["id"]] = e
        if e.tag == "sentinel":
            return ""
        text = e.text or ""
        children = list(e)
        if children and not text.strip():
            return "|".join(val(c) for c in children)
        return text if text else e.attrib.get("fmt", "")

    out = []
    for r in node.findall("row"):
        out.append(dict(zip(cols, [val(e) for e in r])))
    return cols, out


def main():
    args = sys.argv[1:]
    trace = args[0]
    min_ms = 1.0
    listing = "--list" in args
    if "--min-ms" in args:
        min_ms = float(args[args.index("--min-ms") + 1])
    _, gi = rows(trace, "metal-gpu-intervals")
    comp = [(int(r["start"]), int(r["duration"]), r.get("process", ""), r["event-label"].split("|")[0]) for r in gi if r["channel-name"] == "Compute"]
    comp = [c for c in comp if c[1] >= min_ms * 1e6]
    comp.sort()
    if not comp:
        print("no Compute interval")
        return
    t0, t1 = comp[0][0], max(s + d for s, d, _, _ in comp)
    busy = sum(d for _, d, _, _ in comp)
    gaps = [comp[i + 1][0] - (comp[i][0] + comp[i][1]) for i in range(len(comp) - 1)]
    print(f"compute intervals ≥ {min_ms} ms: {len(comp)}, busy {busy / 1e6:.1f} ms over a window of {(t1 - t0) / 1e6:.1f} ms ({100 * busy / (t1 - t0):.1f} %), gaps: {len([g for g in gaps if g > 0])} positive, total {sum(g for g in gaps if g > 0) / 1e6:.1f} ms, median {sorted(gaps)[len(gaps) // 2] / 1e6:.3f} ms" if gaps else f"compute intervals: 1, {busy / 1e6:.1f} ms")
    _, ps = rows(trace, "gpu-performance-state-intervals")
    states = [(int(r["start"]), int(r["duration"]), r["gpu-performance-state"]) for r in ps]
    states.sort()

    def share(a, b):
        tot = {}
        for s, d, st in states:
            lo, hi = max(s, a), min(s + d, b)
            if hi > lo:
                tot[st] = tot.get(st, 0) + hi - lo
        covered = sum(tot.values())
        return {k: v / (b - a) for k, v in tot.items()}, covered / (b - a) if b > a else 0

    sh, cov = share(t0, t1)
    print("performance states over the window (share of time; the rest unrecorded = the state before): " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in sorted(sh.items())) + f"; recorded {100 * cov:.0f} %")
    if listing:
        print("start (s)\tduration (ms)\tMax share\tMedium\tMinimum\tprocess\tlabel")
        for s, d, p, lab in comp:
            sh, _ = share(s, s + d)
            print(f"{(s - t0) / 1e9:.3f}\t{d / 1e6:.1f}\t{100 * sh.get('Maximum', 0):.0f} %\t{100 * sh.get('Medium', 0):.0f} %\t{100 * sh.get('Minimum', 0):.0f} %\t{p}\t{lab[:50]}")


if __name__ == "__main__":
    main()
