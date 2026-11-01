"""The mlx-lm bf16 reference of a Qwen3.5 / Qwen3.6 / Qwen3.8 checkpoint.

usage: python3 -I ref_qwen.py MODEL_DIR NAME OUT_DIR

The prompts of compare.py (`short`, `long`) in herbert-metal-chat's template
with `--no-system` and the thinking open (`<|im_start|>assistant\\n<think>\\n`),
tokenized by the model's tokenizer.json; 48 greedy ids from the prompt cache,
the end of turn kept. Writes OUT_DIR/ref-NAME.json.
"""
import json, os, sys
from pathlib import Path
import mlx.core as mx
from mlx_lm.utils import load_model, load_tokenizer
from mlx_lm.models.cache import make_prompt_cache
from tokenizers import Tokenizer

path, name, out_dir = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
here = os.path.dirname(os.path.abspath(__file__))
model, _cfg = load_model(path, strict=False)
tok = load_tokenizer(path)
hf = Tokenizer.from_file(str(path / "tokenizer.json"))
stops = [hf.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>")]
long = open(os.path.join(here, "long-prompt.txt")).read()
out = {"model": str(path), "stops": stops}
for key, text in [("short", "Bonjour, que fais-tu ?"), ("long", long)]:
    prompt = "<|im_start|>user\n" + text.strip() + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    ids = hf.encode(prompt, add_special_tokens=False).ids
    cache = make_prompt_cache(model)
    y = mx.array([ids])
    gen, tops = [], []
    for i in range(48):
        lg = model(y, cache=cache)[0, -1].astype(mx.float32)
        mx.eval(lg)
        top = mx.argsort(lg)[-5:][::-1].tolist()
        tops.append([[t, float(lg[t])] for t in top])
        t = top[0]
        gen.append(t)
        if t in stops:
            break
        y = mx.array([[t]])
    out[key] = {"prompt_ids": ids, "gen_ids": gen, "top5": tops, "gen_text": tok.decode(gen)}
    print(name, key, len(ids), gen[:12], repr(tok.decode(gen)[:200]), flush=True)
os.makedirs(out_dir, exist_ok=True)
json.dump(out, open(os.path.join(out_dir, f"ref-{name}.json"), "w"), ensure_ascii=False)
