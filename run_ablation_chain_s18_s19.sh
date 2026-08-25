#!/bin/bash
# Auto-chaining ablation: S18 → S19 (then S17 → S20)
set -e

export OLLAMA_HOST="http://localhost:11435"
cd /home/xxy/vrx_ws/src/marine_env

LOG_DIR="/home/xxy/vrx_ws/src/marine_env/ablation_logs"
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/chain_s18_s19.log"
}

run_group() {
    local scenario=$1
    local groups=$2
    local logfile="$LOG_DIR/s${scenario}_$(echo $groups | tr ',' '_').log"
    log "STARTING: Scenario $scenario Groups $groups"
    python3 run_ablation.py \
        --groups "$groups" \
        --scenarios "$scenario" \
        --repeats-full 20 \
        --resume \
        2>&1 | tee "$logfile"
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        log "COMPLETED: Scenario $scenario Groups $groups"
    else
        log "FAILED (exit=$exit_code): Scenario $scenario Groups $groups"
        return $exit_code
    fi
}

verify_output() {
    local scenario=$1
    local group=$2
    local dir="ablation_output/scenario_${scenario}/group_${group}"
    local summary="$dir/summary.json"
    if [ ! -f "$summary" ]; then
        log "VERIFY FAIL: $summary missing"
        return 1
    fi
    local runs=$(python3 -c "import json; d=json.load(open('$summary')); print(d.get('num_runs',0))" 2>/dev/null)
    local collision=$(python3 -c "import json; d=json.load(open('$summary')); print(d.get('collision_rate',-1))" 2>/dev/null)
    log "VERIFY: S${scenario}_${group} runs=$runs collision_rate=$collision"
    return 0
}

log "========== ABLATION CHAIN S18-S19 START =========="

# Phase 1: S18 (三船交错交叉, Rule 15/16/17)
for g in A B D; do
    if [ -f "ablation_output/scenario_18/group_${g}/summary.json" ]; then
        log "S18_${g} already completed — skipping"
    else
        run_group 18 $g
        verify_output 18 $g
    fi
done
log "=== S18 COMPLETE ==="
python3 generate_comparison_charts.py 18 2>&1 | tee -a "$LOG_DIR/chain_s18_s19.log"

# Phase 2: S19 (夜航灯光, Rules 20-23)
for g in A B D; do
    if [ -f "ablation_output/scenario_19/group_${g}/summary.json" ]; then
        log "S19_${g} already completed — skipping"
    else
        run_group 19 $g
        verify_output 19 $g
    fi
done
log "=== S19 COMPLETE ==="
python3 generate_comparison_charts.py 19 2>&1 | tee -a "$LOG_DIR/chain_s18_s19.log"

log "========== ABLATION CHAIN S18-S19 DONE =========="
