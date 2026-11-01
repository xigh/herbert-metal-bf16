#!/usr/bin/env python3
"""The tables of the article #004 campaign (bench/f16-campaign.sh logs): per length and model, the ppN of herbert
(h-bf16 / h-f16: the `gpu t/s` table's last row for the test, which is the GPU rate), llama-bench (the ±-row of the
test, bf16 or f16 GGUF) and mlx_lm benchmark (the Averages line); tgN alike (DEPTH runs: the `tg128 @ dN` rows).
usage: bench/art004_tables.py LOGS_DIR [pp|tg]   (LOGS_DIR holds ppN/MODEL/ENGINE.log or MODEL/ENGINE.log)
"""
import re, sys
from pathlib import Path

def herbert(text, test):
    rows = re.findall(r"\| \S+ \| herbert-metal-\S+ \| " + re.escape(test) + r" \| ([0-9.]+) ±", text)
    return float(rows[-1]) if rows else None  # the last table is the GPU rate

def llama(text, test):
    rows = re.findall(r"\|\s+" + re.escape(test) + r"\s+\|\s+([0-9.]+) ±", text)
    return float(rows[-1]) if rows else None

def mlx(text, test):
    m = re.search(r"Averages:\s+prompt_tps=([0-9.]+), generation_tps=([0-9.]+)", text)
    if not m:
        return None
    return float(m.group(1)) if test.startswith("pp") else float(m.group(2))

def cell(x):
    return f"{x:,.0f}".replace(",", " ") if x else "—"

def main():
    root = Path(sys.argv[1]); kind = sys.argv[2] if len(sys.argv) > 2 else "pp"
    groups = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("pp")], key=lambda p: int(p.name[2:])) or [root]
    for g in groups:
        models = sorted([m for m in g.iterdir() if m.is_dir()], key=lambda m: m.name)
        engines = sorted({f.stem for m in models for f in m.glob("*.log")})
        test = g.name if g is not root and kind == "pp" else None
        print(f"### {g.name if g is not root else root.name}")
        print("| model | test | " + " | ".join(engines) + " |")
        print("| --- | --- | " + " | ".join("---:" for _ in engines) + " |")
        for m in models:
            texts = {e: (m / f"{e}.log").read_text(errors="replace") if (m / f"{e}.log").exists() else "" for e in engines}
            tests = [test] if test else sorted({t for t in re.findall(r"\| (pp\d+|tg\d+(?: @ d\d+)?) \|", texts.get("h-f16", "") + texts.get("h-bf16", ""))})
            if kind == "tg":
                tests = [t for t in tests if t.startswith("tg")]  # the depth runs' pp1 row is the prompt's single token
            for t in tests:
                vals = []
                for e in engines:
                    f = herbert if e.startswith("h-") else llama if e.startswith("llama") else mlx
                    vals.append(f(texts[e], t))
                print(f"| {m.name} | {t} | " + " | ".join(cell(v) for v in vals) + " |")
        print()

if __name__ == "__main__":
    main()
