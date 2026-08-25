#!/bin/bash
# Auto-chaining ablation for S17 → S18 → S19 → S20
set -e

export OLLAMA_HOST="http://localhost:11435"
cd /home/xxy/vrx_ws/src/marine_env

LOG_DIR="/home/xxy/vrx_ws/src/marine_env/ablation_logs"
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/chain_s17_s20.log"
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

log "========== ABLATION CHAIN S17-S20 START =========="

for scenario in 17 18 19 20; do
    for g in A B D; do
        if [ -f "ablation_output/scenario_${scenario}/group_${g}/summary.json" ]; then
            log "S${scenario}_${g} already completed — skipping"
        else
            run_group $scenario $g
            verify_output $scenario $g
        fi
    done
    log "=== S${scenario} COMPLETE ==="
    # Generate cross-group comparison charts
    python3 generate_comparison_charts.py $scenario 2>&1 | tee -a "$LOG_DIR/chain_s17_s20.log"
done

log "========== ABLATION CHAIN S17-S20 DONE =========="
