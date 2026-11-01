"""Greedy ids of herbert-metal-chat against the mlx-lm bf16 reference.

usage: python3 -I bench/compare.py BINARY MODELS_DIR [E2B E4B Qwen3.5-0.8B …] [--log DIR] [--ref f16] [-- OPTIONS OF THE CHAT]

For each model and each reference prompt (`short`, `long` of
bench/reference/ref-<name>.json, made by ref_gemma.py or ref_qwen.py; a Gemma
model is MODELS_DIR/gemma-4-<name>-it-qat-q4_0-unquantized, a Qwen one
MODELS_DIR/<name> or MODELS_DIR/qwen/<name>), runs the chat with
`--no-system`, checks that the prompt ids are the reference's, and prints how
many generated ids are identical before the first difference (the comparison
stops at the reference's end of turn). The full stderr of every run is kept
in the log directory. `--ref f16`: against the mlx-lm float16 references
(ref-<name>-f16.json, made by ref_f16.py) instead of the bf16 ones.
"""
import json, os, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:]]
extra = []
if "--" in args:
    i = args.index("--")
    args, extra = args[:i], args[i + 1:]
log_dir = None
if "--log" in args:
    i = args.index("--log")
    log_dir = args[i + 1]
    del args[i:i + 2]
suffix = ""
if "--ref" in args:
    i = args.index("--ref")
    suffix = "" if args[i + 1] == "bf16" else "-" + args[i + 1]
    del args[i:i + 2]
binary, models = args[0], args[1]
names = args[2:] or ["E2B", "E4B"]
texts = {"short": "Bonjour, que fais-tu ?", "long": open(os.path.join(here, "reference", "long-prompt.txt")).read()}
if log_dir:
    os.makedirs(log_dir, exist_ok=True)
worst = 0
for name in names:
    ref = json.load(open(os.path.join(here, "reference", f"ref-{name}{suffix}.json")))
    if name.startswith("Qwen"):
        model = next((p for p in (os.path.join(models, name), os.path.join(models, "qwen", name)) if os.path.isdir(p)), os.path.join(models, name))
        stops = ref["stops"]
    else:
        model = os.path.join(models, f"gemma-4-{name}-it-qat-q4_0-unquantized")
        stops = [1, 106]
    for key in ["short", "long"]:
        want = ref[key]["gen_ids"]
        cmd = [binary, model, "--no-system", "--max-tokens", str(len(want) + 1), "--ask", texts[key]] + extra
        r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "HERBERT_METAL_PROMPT_IDS": "1", "HERBERT_METAL_TOP": "5"})
        if log_dir:
            with open(os.path.join(log_dir, f"{name}-{key}.log"), "w") as f:
                f.write(" ".join(cmd) + "\n" + r.stdout + "\n--- stderr ---\n" + r.stderr)
        lines = {l.split("]", 1)[0] + "]": l.split("]", 1)[1].strip() for l in r.stderr.splitlines() if l.startswith("[")}
        if r.returncode != 0 or "[ids]" not in lines:
            print(f"{name} {key}: run failed ({r.returncode})\n{r.stderr[-2000:]}")
            worst = 2
            continue
        prompt = [int(t) for t in lines["[prompt]"].split()]
        got = [int(t) for t in lines["[ids]"].split()]
        # The reference keeps its end-of-turn token; the chat stops before it.
        want_cmp = [t for t in want if t not in stops]
        same = 0
        while same < min(len(got), len(want_cmp)) and got[same] == want_cmp[same]:
            same += 1
        prompt_ok = prompt == ref[key]["prompt_ids"]
        verdict = "identical" if same == len(want_cmp) and len(got) == len(want_cmp) else f"first difference at {same}"
        print(f"{name} {key}: prompt {'=' if prompt_ok else '≠'} reference ({len(prompt)} ids) · {same}/{len(want_cmp)} ids identical · {verdict}")
        if same < len(want_cmp):
            print(f"    reference {want_cmp[same:same + 8]}\n    metal     {got[same:same + 8]}")
            # The margins at the first difference: the reference's top-5 and ours (HERBERT_METAL_TOP).
            fmt = lambda top: " ".join(f"{t}:{l:.4g}" for t, l in top)
            top5 = ref[key].get("top5", [])  # per step (ref_qwen.py, ref_f16.py); ref_gemma.py's bf16 ones have none
            rt = top5[same] if same < len(top5) else []
            if rt:
                print(f"    reference top-5 at {same}: {fmt(rt)} (margin {rt[0][1] - rt[1][1]:.4g})")
            ours = [l for l in r.stderr.splitlines() if l.startswith(f"[top {same}]")]
            if ours:
                mt = [(int(a), float(b)) for a, b in (x.split(":") for x in ours[0].split("]", 1)[1].split())]
                print(f"    metal top-5 at {same}: {fmt(mt)} (margin {mt[0][1] - mt[1][1]:.4g})")
        if not prompt_ok:
            worst = max(worst, 1)
sys.exit(worst)
