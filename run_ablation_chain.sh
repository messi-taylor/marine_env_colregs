#!/bin/bash
# Auto-chaining ablation experiment script
# Chains: S14_B → S14_D → S15_ABD → S16_ABD
set -e

export OLLAMA_HOST="http://localhost:11435"
cd /home/xxy/vrx_ws/src/marine_env

LOG_DIR="/home/xxy/vrx_ws/src/marine_env/ablation_logs"
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/chain.log"
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
    local metrics="$dir/metrics.csv"

    if [ ! -f "$summary" ]; then
        log "VERIFY FAIL: $summary missing"
        return 1
    fi
    if [ ! -f "$metrics" ]; then
        log "VERIFY FAIL: $metrics missing"
        return 1
    fi

    local runs=$(python3 -c "import json; d=json.load(open('$summary')); print(d.get('num_runs',0))" 2>/dev/null)
    local collision=$(python3 -c "import json; d=json.load(open('$summary')); print(d.get('collision_rate',-1))" 2>/dev/null)

    log "VERIFY: S${scenario}_${group} runs=$runs collision_rate=$collision"
    return 0
}

# ── Main chain ──

log "========== ABLATION CHAIN START =========="

# Phase 1: S14 Group B
if [ -f "ablation_output/scenario_14/group_B/summary.json" ]; then
    log "S14_B already completed — skipping"
else
    run_group 14 B
    verify_output 14 B
fi

# Phase 2: S14 Group D
if [ -f "ablation_output/scenario_14/group_D/summary.json" ]; then
    log "S14_D already completed — skipping"
else
    run_group 14 D
    verify_output 14 D
fi

log "=== S14 COMPLETE ==="

# Phase 3: S15 Groups A,B,D
for g in A B D; do
    if [ -f "ablation_output/scenario_15/group_${g}/summary.json" ]; then
        log "S15_${g} already completed — skipping"
    else
        run_group 15 "$g"
        verify_output 15 "$g"
    fi
done

log "=== S15 COMPLETE ==="

# Phase 4: S16 Groups A,B,D
for g in A B D; do
    if [ -f "ablation_output/scenario_16/group_${g}/summary.json" ]; then
        log "S16_${g} already completed — skipping"
    else
        run_group 16 "$g"
        verify_output 16 "$g"
    fi
done

log "=== S16 COMPLETE ==="
log "========== ABLATION CHAIN DONE =========="
