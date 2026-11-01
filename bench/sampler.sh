#!/bin/bash
# The state of the machine every PERIOD seconds (default 30) until killed: GPU counters (ioreg), thermal and power notes
# (pmset), memory pressure, swap, vm_stat (cumulative pages: subtract two samples), the herbert processes, the top of the
# CPU. Every command has a timeout: a stuck one does not stop the sampling.
# usage: bench/sampler.sh OUTFILE [PERIOD]
OUT=$1; PERIOD=${2:-30}
t() { perl -e 'alarm shift; exec @ARGV' 10 "$@" 2>&1; }
while true; do
  {
    echo "== $(date '+%Y-%m-%d %H:%M:%S') uptime-raw $(perl -MTime::HiRes=clock_gettime -e 'print clock_gettime(8)' 2>/dev/null)"
    t ioreg -r -c IOAccelerator -d 1 | grep -o '"\(Device\|Renderer\|Tiler\) Utilization %"=[0-9]*\|"In use system memory"=[0-9]*\|"Alloc system memory"=[0-9]*\|"recoveryCount"=[0-9]*\|"SplitSceneCount"=[0-9]*' | tr '\n' ' '; echo
    t pmset -g therm | grep -v '^Note: No'
    t pmset -g batt | head -1
    t memory_pressure | tail -1
    echo "swap $(t sysctl -n vm.swapusage)"
    t vm_stat | grep -E 'Pages (free|active|inactive|speculative|wired down|occupied by compressor)|Pageins|Pageouts|Swapins|Swapouts|Decompressions|Compressions|File-backed|Anonymous' | tr -s ' ' | tr '\n' ';'; echo
    t ps -eo pid,etime,%cpu,rss,state,command | grep '[h]erbert-metal' | cut -c1-140
    t ps -Ao pid,%cpu,rss,comm -r | sed -n 2,6p | cut -c1-110
  } >> "$OUT"
  sleep "$PERIOD"
done
