"""The mlx-lm bf16 reference of a Gemma-4 checkpoint.

usage: python3 -I ref_gemma.py MODEL_DIR NAME OUT_DIR

The prompts of compare.py (`short`, `long`), 48 greedy ids. Writes OUT_DIR/ref-NAME.json.
"""
import os, sys, json, subprocess
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
path, name, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
here = os.path.dirname(os.path.abspath(__file__))
from mlx_lm.utils import load_model, load_tokenizer
from pathlib import Path
path = Path(path)
model, _cfg = load_model(path, strict=False)
tok = load_tokenizer(path)
long = open(os.path.join(here, "long-prompt.txt")).read()
out = {}
for key, text in [("short", "Bonjour, que fais-tu ?"), ("long", long)]:
    from tokenizers import Tokenizer
    ids = Tokenizer.from_file(str(path / "tokenizer.json")).encode("<bos><|turn>user\n" + text + "<turn|>\n<|turn>model\n", add_special_tokens=False).ids
    x = mx.array([ids])
    logits = model(x)
    last = logits[0, -1].astype(mx.float32)
    mx.eval(last)
    top = mx.argsort(last)[-5:][::-1].tolist()
    gen = []
    cur = list(ids)
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    y = mx.array([ids])
    for i in range(48):
        lg = model(y, cache=cache)[:, -1, :]
        t = int(mx.argmax(lg, axis=-1).item())
        gen.append(t)
        if t in (1, 106): break
        y = mx.array([[t]])
    out[key] = {"prompt_ids": ids, "top5_last": top, "top5_logits": [float(last[i]) for i in top], "gen_ids": gen, "gen_text": tok.decode(gen)}
    print(key, len(ids), top, tok.decode(gen)[:300])
os.makedirs(out_dir, exist_ok=True)
json.dump(out, open(os.path.join(out_dir, f"ref-{name}.json"), "w"), ensure_ascii=False)
