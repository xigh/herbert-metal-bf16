#!/bin/zsh
# One model of the f16 comparison: ppN for each N of PP, and tg128, by each engine in the order given, a pause between two engines,
# every output kept; the protocol of the earlier bf16 campaigns with this tree's two engines, the f16 one and
# its bf16 witness, and the GGUF in bf16 and in f16. The same script on the M3 Ultra, the M1 Air and the M5: the paths by env.
#
# usage: bench/f16-campaign.sh OUT_DIR NAME ENGINE...
#   NAME    E2B | E4B | Qwen3.5-0.8B | Qwen3.5-2B | Qwen3.5-4B | Qwen3.5-9B | Qwen3.8-27B | Qwen3.6-35B-A3B
#   ENGINE  h-f16 (this tree --features f16, target/f16) | h-bf16 (this tree, target/release) | llama-bf16 |
#           llama-f16 (llama-bench on the GGUF) | mlx (mlx_lm benchmark on the checkpoint, bf16) | rapid (Rapid-MLX
#           through bench/rapid_client.py) | basert (BaseRT on a bf16 .base converted in BASEDIR, dropped after)
# env: PAUSE (seconds, default 30), REPS (default 3), PP (prompt lengths, comma separated, default 512), TG (default
#      128; herbert and llama-bench take the list in one process, mlx-lm, Rapid-MLX and BaseRT one process or request
#      per length, tgTG with the prompt of 512 tokens, or the first, or TGP, and tg8, tg8, tg0 with the others),
#      DEPTH (default 0; > 0: tgTG only, after DEPTH tokens of context: herbert -d, llama-bench -p 0 -d,
#      mlx_lm benchmark -p DEPTH, whose generation follows its prompt; rapid and basert are not run),
#      LLAMA (llama-bench's dir, its libraries there, default ~/llama.cpp/build/bin),
#      GGUF (default ~/gguf: <NAME>-BF16.gguf, <NAME>-F16.gguf, gemma-4-<NAME>-bf16.gguf),
#      MLX (mlx-lm's python, default ~/mlx-bench-venv/bin/python), EN (default ~/engines), MODELS (default ~/models),
#      W (warm-up runs, default 1; 0: none, llama-bench --no-warmup; mlx_lm benchmark always runs one),
#      BASEDIR (default ~/basert-tmp), QUIET (a CPU %: wait, up to 10 min, until no process uses more, the waits logged;
#      the M5's rule), QUIET_IGNORE (an extended regex: the processes QUIET does not wait for,
#      still logged before and after each run; a background copy on the M5), COOL (non-empty, the default on the M5: wait, up to
#      10 min, until the thermal pressure level is back to 0 before each engine, the waits logged; COOL= : no wait)
#
# Same protocol for all: ppN from an empty context on pseudo-random ids (llama-bench's random tokens, BaseRT
# --fill random, mlx_lm benchmark's random ints, herbert-metal-bench's LCG, Rapid-MLX's decoded LCG text), tgN from an
# (almost) empty context, one warm-up, REPS timed runs. A missing input is written in the log and the engine skipped.
set -u
out=$1; name=$2; shift 2
PAUSE=${PAUSE:-30}; REPS=${REPS:-3}; W=${W:-1}; PP=${PP:-512}; TG=${TG:-128}; DEPTH=${DEPTH:-0}
HB=${0:A:h}/..
LLAMA=${LLAMA:-$HOME/llama.cpp/build/bin}
GGUF=${GGUF:-$HOME/gguf}
MLX=${MLX:-$HOME/mlx-bench-venv/bin/python}
EN=${EN:-$HOME/engines}
MODELS=${MODELS:-$HOME/models}
BASEDIR=${BASEDIR:-$HOME/basert-tmp}
QUIET=${QUIET:-}; QUIET_IGNORE=${QUIET_IGNORE:-}
lw=(); [[ $W == 0 ]] && lw=(--no-warmup)
pps=(${(s:,:)PP})
tgp=$pps[1]; (( ${pps[(Ie)512]} )) && tgp=512; tgp=${TGP:-$tgp}
case $name in
  E2B|E4B) dir=$MODELS/gemma-4-$name-it-qat-q4_0-unquantized; gb=$GGUF/gemma-4-$name-bf16.gguf; gf=$GGUF/gemma-4-$name-f16.gguf ;;
  Qwen3.8-27B|Qwen3.6-35B-A3B) dir=$MODELS/$name; gb=$GGUF/$name-BF16.gguf; gf=$GGUF/$name-F16.gguf ;;
  *) dir=$MODELS/qwen/$name; gb=$GGUF/$name-BF16.gguf; gf=$GGUF/$name-F16.gguf ;;
esac
base=$BASEDIR/$name-bf16.base
mkdir -p $out/$name
first=1
for e in "$@"; do
  [[ $DEPTH -gt 0 && ( $e == rapid || $e == basert ) ]] && continue
  [[ $first == 1 ]] || { echo "pause $PAUSE s"; sleep $PAUSE; }
  first=0
  log=$out/$name/$e.log
  echo "=== $name $e $(date '+%F %T')" | tee $log
  if [[ -n $QUIET ]]; then
    for i in {1..20}; do
      busy=$(ps -axo pcpu,comm | awk -v q=$QUIET 'NR>1 && $1>q' | grep -vE "${QUIET_IGNORE:-^$}")
      [[ -z $busy ]] && break
      { echo "busy $(date +%T), waiting 30 s:"; echo $busy; } >> $log
      sleep 30
    done
  fi
  if [[ -n ${COOL-$([[ $(sysctl -n machdep.cpu.brand_string) == *M5* ]] && echo 1)} ]]; then
    for i in {1..60}; do
      lvl=$(notifyutil -g com.apple.system.thermalpressurelevel 2>/dev/null | awk '{print $2}')
      [[ -z $lvl || $lvl == 0 ]] && break
      echo "thermal pressure $lvl $(date +%T), waiting 10 s" >> $log
      sleep 10
    done
  fi
  { pmset -g therm; notifyutil -g com.apple.system.thermalpressurelevel; ps -axo pcpu,comm | awk 'NR>1 && $1>20'; vm_stat | grep -E "Swapouts|Pages free"; } >> $log 2>&1
  input=$dir
  case $e in llama-bf16) input=$gb ;; llama-f16) input=$gf ;; esac
  echo "input $input" >> $log
  if [[ ! -e $input ]]; then
    echo "missing input $input: not run" | tee -a $log
    continue
  fi
  case $e in
    h-f16|h-bf16)
      bin=$HB/target/release/herbert-metal-bench; [[ $e == h-f16 ]] && bin=$HB/target/f16/release/herbert-metal-bench
      if [[ $DEPTH -gt 0 ]]; then
        $bin $dir -p 1 -n $TG -d $DEPTH -r $REPS -w $W >> $log 2>&1
      else
        $bin $dir -p $PP -n $TG -r $REPS -w $W >> $log 2>&1
      fi ;;
    llama-bf16|llama-f16)
      if [[ $DEPTH -gt 0 ]]; then
        DYLD_LIBRARY_PATH=$LLAMA $LLAMA/llama-bench -m $input -p 0 -n $TG -d $DEPTH -r $REPS $lw -o md >> $log 2>&1
      else
        DYLD_LIBRARY_PATH=$LLAMA $LLAMA/llama-bench -m $input -p $PP -n $TG -r $REPS $lw -o md >> $log 2>&1
      fi ;;
    mlx)
      if [[ $DEPTH -gt 0 ]]; then
        $MLX -m mlx_lm benchmark --model $dir -p $DEPTH -g $TG -n $REPS >> $log 2>&1
      else
        for p in $pps; do
          echo "--- pp$p" >> $log
          $MLX -m mlx_lm benchmark --model $dir -p $p -g $(( p == tgp ? TG : 8 )) -n $REPS >> $log 2>&1
        done
      fi ;;
    rapid)
      RAPID_MLX_TELEMETRY=0 HF_HUB_OFFLINE=1 $EN/rapid-mlx/venv/bin/rapid-mlx serve $dir --disable-model-downloads --no-mllm \
        --disable-prefix-cache --disable-disk-caches --port 8765 > $out/$name/rapid-serve.log 2>&1 &
      pid=$!
      up=0
      for i in {1..600}; do
        curl -sf http://localhost:8765/v1/models > /dev/null && { up=1; break; }
        kill -0 $pid 2> /dev/null || break
        sleep 1
      done
      if [[ $up == 1 ]]; then
        for p in $pps; do
          echo "--- pp$p" >> $log
          $EN/rapid-mlx/venv/bin/python -I $HB/bench/rapid_client.py $dir 8765 -p $p -n $(( p == tgp ? TG : 8 )) -r $REPS -w $W >> $log 2>&1
        done
      else
        echo "rapid-mlx did not come up" >> $log
      fi
      kill $pid 2> /dev/null; wait $pid 2> /dev/null ;;
    basert)
      mkdir -p $BASEDIR
      if [[ ! -f $base ]]; then
        df -h $BASEDIR | tail -1 >> $out/$name/basert-convert.log
        (cd $EN/baseRT && ./base-convert/target/release/basert convert $dir --target bf16 -o $base) >> $out/$name/basert-convert.log 2>&1
      fi
      rc=0
      for p in $pps; do
        echo "--- pp$p" >> $log
        (cd $EN/baseRT && ./build/basert-bench $base -p $p -n $(( p == tgp ? TG : 0 )) -r $REPS -w $W --fill random) >> $log 2>&1 || rc=$?
      done
      rm -f $base && echo "removed $base" >> $log
      (exit $rc) ;;
  esac
  echo "exit $? $(date '+%F %T')" >> $log
  { pmset -g therm; notifyutil -g com.apple.system.thermalpressurelevel; vm_stat | grep -E "Swapouts|Pages free"; } >> $log 2>&1
  grep -E "^\||prompt_tps|generation_tps|tok/s" $log | grep -v autotune | tail -4
done
