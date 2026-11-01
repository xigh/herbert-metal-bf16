"""The mlx-lm reference of a Qwen3.5 / 3.6 / 3.8 or Gemma-4 checkpoint, its
bf16 parameters cast to another dtype after loading.

usage: python3 -I ref_f16.py MODEL_DIR NAME OUT_DIR [--dtype float16|bfloat16]

The model is loaded lazily, every bf16 parameter cast to `--dtype` (default
float16; the f32 ones stay f32), then evaluated one parameter at a time; the
Counter of the parameter dtypes before and after is printed and written. The
prompts and the greedy decoding are those of ref_qwen.py (Qwen: the thinking open, 48 ids from the
prompt cache, the end of turn kept) and ref_gemma.py (Gemma: `<bos>` and the
turn markers, 48 ids, stops 1 and 106, `top5_last` from the whole prompt; the
argmax of each family's script, which part equal logits differently), with
the five best logits of every generated position for both families. Writes
OUT_DIR/ref-NAME-f16.json (or ref-NAME-bf16.json with `--dtype bfloat16`, to
check the script against the reference already in bench/reference).
"""
import collections, json, os, sys
from pathlib import Path
import mlx.core as mx
import mlx.utils
from mlx_lm.utils import load_model, load_tokenizer
from mlx_lm.models.cache import make_prompt_cache
from tokenizers import Tokenizer

args = sys.argv[1:]
dtype_name = "float16"
if "--dtype" in args:
    i = args.index("--dtype")
    dtype_name = args[i + 1]
    del args[i:i + 2]
path, name, out_dir = Path(args[0]), args[1], args[2]
dtype = {"float16": mx.float16, "bfloat16": mx.bfloat16}[dtype_name]
suffix = {"float16": "f16", "bfloat16": "bf16"}[dtype_name]
here = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(path / "config.json"))
model_type = cfg.get("text_config", {}).get("model_type") or cfg.get("model_type", "")
gemma = model_type.startswith("gemma")

model, _cfg = load_model(path, lazy=True, strict=False)


def counter():
    return collections.Counter(str(v.dtype) for _, v in mlx.utils.tree_flatten(model.parameters()))


before = counter()
model.set_dtype(dtype, predicate=lambda d: d == mx.bfloat16)
# One parameter at a time: a single evaluation of the whole cast model (the
# 35B, 71 GB) exceeds the GPU's command buffer timeout.
for _, v in mlx.utils.tree_flatten(model.parameters()):
    mx.eval(v)
after = counter()
print(name, dtype_name, "parameters before", dict(before), "after", dict(after), flush=True)

tok = load_tokenizer(path)
hf = Tokenizer.from_file(str(path / "tokenizer.json"))
long = open(os.path.join(here, "long-prompt.txt")).read()
if gemma:
    stops = [1, 106]
else:
    stops = [hf.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>")]
out = {"model": str(path), "dtype": dtype_name, "params_before": dict(before), "params_after": dict(after), "stops": stops}


def top5(lg):
    top = mx.argsort(lg)[-5:][::-1].tolist()
    return top, [[t, float(lg[t])] for t in top]


for key, text in [("short", "Bonjour, que fais-tu ?"), ("long", long)]:
    if gemma:
        prompt = "<bos><|turn>user\n" + text + "<turn|>\n<|turn>model\n"
    else:
        prompt = "<|im_start|>user\n" + text.strip() + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    ids = hf.encode(prompt, add_special_tokens=False).ids
    entry = {"prompt_ids": ids}
    if gemma:
        last = model(mx.array([ids]))[0, -1].astype(mx.float32)
        mx.eval(last)
        top, _ = top5(last)
        entry["top5_last"] = top
        entry["top5_logits"] = [float(last[t]) for t in top]
    cache = make_prompt_cache(model)
    y = mx.array([ids])
    gen, tops = [], []
    for i in range(48):
        lg = model(y, cache=cache)[0, -1].astype(mx.float32)
        mx.eval(lg)
        top, pairs = top5(lg)
        if any(v != v for _, v in pairs):
            print(name, key, "NaN in the logits at", i, flush=True)
        tops.append(pairs)
        # Gemma: ref_gemma.py's argmax (the lowest id among equal logits);
        # Qwen: ref_qwen.py's argsort (the highest).
        t = int(mx.argmax(lg).item()) if gemma else top[0]
        gen.append(t)
        if t in stops:
            break
        y = mx.array([[t]])
    entry.update({"gen_ids": gen, "top5": tops, "gen_text": tok.decode(gen)})
    out[key] = entry
    print(name, key, len(ids), gen[:12], repr(tok.decode(gen)[:200]), flush=True)
os.makedirs(out_dir, exist_ok=True)
json.dump(out, open(os.path.join(out_dir, f"ref-{name}-{suffix}.json"), "w"), ensure_ascii=False)
