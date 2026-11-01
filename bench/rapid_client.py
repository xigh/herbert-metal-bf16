"""ppN and tgN of a running Rapid-MLX server, as llama-bench measures them.

usage: python3 -I rapid_client.py MODEL_DIR PORT [-p 512] [-n 128] [-r 3] [-w 1]

Rapid-MLX's own benchmark refuses a local path, and its OpenAI API takes
text: the prompt of a ppN run is N pseudo-random ids (a different seed per
run, so no prefix is shared) decoded to text and re-tokenized, cut at N ids;
`/v1/completions` with `max_tokens 1`, the server's `time_to_first_token_ms`
and `usage.prompt_tokens` give the rate. A tgN run asks for N tokens of a
story from a short raw prompt (no chat template); the rate is
`1000 / mean_itl_ms` (the server's mean inter-token latency), and the run is
flagged when fewer than N tokens came back. One line per run, then the
llama-bench table (mean ± standard deviation).
"""
import json, statistics, sys, time, urllib.request
from tokenizers import Tokenizer

args = sys.argv[1:]
model, port = args[0], int(args[1])
opt = {"-p": 512, "-n": 128, "-r": 3, "-w": 1}
for i in range(2, len(args), 2):
    opt[args[i]] = int(args[i + 1])
pp, tg, reps, warm = opt["-p"], opt["-n"], opt["-r"], opt["-w"]
tok = Tokenizer.from_file(f"{model}/tokenizer.json")
url = f"http://localhost:{port}/v1/completions"


def lcg(n, seed):
    x = (0x2545F4914F6CDD1D + seed * 0x9E3779B97F4A7C15) & (2**64 - 1)
    out = []
    for _ in range(n):
        x = (x * 6364136223846793005 + 1442695040888963407) & (2**64 - 1)
        out.append((x >> 33) % 248044)
    return out


def prompt(seed):
    text = tok.decode(lcg(pp, seed))
    ids = tok.encode(text, add_special_tokens=False).ids[:pp]
    return tok.decode(ids)


def post(body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    return out, time.time() - t


name = model.rstrip("/").split("/")[-1]
rows = []
if pp > 0:
    tps = []
    for i in range(warm + reps):
        out, wall = post({"model": model, "prompt": prompt(i + 1), "max_tokens": 1, "temperature": 0.0})
        n, ttft = out["usage"]["prompt_tokens"], out["metrics"]["time_to_first_token_ms"]
        rate = n / (ttft / 1000.0)
        kind = "warm-up" if i < warm else "run"
        print(f"[pp{pp}] {kind} prompt_tokens {n} ttft {ttft:.1f} ms wall {wall:.3f} s {rate:.1f} tok/s", file=sys.stderr, flush=True)
        if i >= warm:
            tps.append(rate)
    rows.append((f"pp{pp}", tps))
if tg > 0:
    tps = []
    story = "<|im_start|>user\nRaconte une très longue histoire de marins, avec beaucoup de détails.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    for i in range(warm + reps):
        out, wall = post({"model": model, "prompt": story, "max_tokens": tg, "temperature": 0.0})
        n, itl = out["usage"]["completion_tokens"], out["metrics"]["mean_itl_ms"]
        rate = 1000.0 / itl
        kind = "warm-up" if i < warm else "run"
        short = "" if n == tg else f" SHORT ({n} < {tg})"
        print(f"[tg{tg}] {kind} completion_tokens {n} mean_itl {itl:.3f} ms wall {wall:.3f} s {rate:.1f} tok/s{short}", file=sys.stderr, flush=True)
        if i >= warm:
            tps.append(rate)
    rows.append((f"tg{tg}", tps))
print("| model | engine | test | t/s |")
print("| --- | --- | ---: | ---: |")
for test, v in rows:
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    print(f"| {name} | rapid-mlx | {test} | {statistics.mean(v):.2f} ± {sd:.2f} |")
