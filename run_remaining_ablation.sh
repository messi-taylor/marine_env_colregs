#!/bin/bash
# Run remaining ablation experiments: S17 A/B/D, S19 D, S20 A/B/D
# Total: 7 groups, estimated ~10 hours

set -e

export OLLAMA_HOST="http://localhost:11435"
cd /home/xxy/vrx_ws/src/marine_env

LOG_DIR="/tmp/ablation_logs"
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/master.log"
}

check_ollama() {
    if curl -s http://localhost:11435/api/tags > /dev/null 2>&1; then
        log "✓ Ollama on 11435 is running"
        return 0
    else
        log "✗ Ollama on 11435 is NOT running!"
        return 1
    fi
}

run_group() {
    local scenario=$1
    local groups=$2
    local logfile="$LOG_DIR/s${scenario}_${groups//,/}.log"

    log "Starting: scenario=$scenario groups=$groups"
    python3 run_ablation.py \
        --groups "$groups" \
        --scenarios "$scenario" \
        --repeats-full 20 \
        --repeats-fast 100 \
        --output ablation_output \
        > "$logfile" 2>&1

    local rc=$?
    if [ $rc -eq 0 ]; then
        log "✓ Completed: scenario=$scenario groups=$groups"
    else
        log "✗ FAILED (rc=$rc): scenario=$scenario groups=$groups"
    fi
    return $rc
}

# ── Main ──
log "=========================================="
log "Starting remaining ablation experiments"
log "7 groups: S17 A,B,D + S19 D + S20 A,B,D"
log "=========================================="

check_ollama || exit 1

# Clean up any invalid previous results
for dir in scenario_17/group_A scenario_17/group_B scenario_17/group_D \
           scenario_19/group_D \
           scenario_20/group_A scenario_20/group_B scenario_20/group_D; do
    if [ -d "ablation_output/$dir" ]; then
        log "Removing invalid previous results: $dir"
        rm -rf "ablation_output/$dir"
    fi
done

# Phase 1: S17 A, B, D (3 groups, ~90-120 min)
log "Phase 1: S17 (加速转向) groups A, B, D"
run_group 17 "A"     # ~30-40 min
run_group 17 "B"     # ~30-40 min
run_group 17 "D"     # ~30-40 min

# Phase 2: S19 D (1 group, ~50 min)
log "Phase 2: S19 (夜航灯光) group D"
run_group 19 "D"     # ~50 min

# Phase 3: S20 A, B, D (3 groups, ~450 min)
log "Phase 3: S20 (六船终极) groups A, B, D"
run_group 20 "A"     # ~150 min
run_group 20 "B"     # ~150 min
run_group 20 "D"     # ~150 min

log "=========================================="
log "ALL ABLATION EXPERIMENTS COMPLETE!"
log "=========================================="
