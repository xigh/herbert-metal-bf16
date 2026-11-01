#!/usr/bin/env python3
"""Rows of one table of an Instruments trace (`xcrun xctrace export --xpath`),
the `ref` attributes resolved to the values they point at, one line per row,
tab separated, the column mnemonics first. Times in ns from the trace's start.

usage: bench/xctrace_table.py TRACE SCHEMA [--run N] [--max N] [--where COL=VALUE]
  e.g. bench/xctrace_table.py runs/trace/try.trace metal-gpu-intervals
"""
import subprocess
import sys
import xml.etree.ElementTree as ET


def export(trace, schema, run):
    xpath = f'/trace-toc/run[@number="{run}"]/data/table[@schema="{schema}"]'
    out = subprocess.run(["xcrun", "xctrace", "export", "--input", trace, "--xpath", xpath], capture_output=True, text=True, check=True).stdout
    return ET.fromstring(out)


def main():
    args = sys.argv[1:]
    trace, schema = args[0], args[1]
    run, limit, where = "1", None, None
    i = 2
    while i < len(args):
        if args[i] == "--run":
            run = args[i + 1]
        elif args[i] == "--max":
            limit = int(args[i + 1])
        elif args[i] == "--where":
            where = args[i + 1].split("=", 1)
        i += 2
    root = export(trace, schema, run)
    node = root.find("node")
    if node is None:
        print(f"no table {schema}", file=sys.stderr)
        return 2
    cols = [c.findtext("mnemonic") for c in node.find("schema").findall("col")]
    byid = {}

    def val(e):
        # An element with an id defines a value; one with a ref repeats it.
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
        return e.attrib.get("fmt", text) if e.tag in ("start-time", "duration") and False else (text if text else e.attrib.get("fmt", ""))

    print("\t".join(cols))
    n = 0
    for row in node.findall("row"):
        vals = [val(e) for e in row]
        if where and (where[0] not in cols or vals[cols.index(where[0])] != where[1]):
            continue
        print("\t".join(vals))
        n += 1
        if limit and n >= limit:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
