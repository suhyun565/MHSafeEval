#!/usr/bin/env bash
# mh_run.sh — End-to-end MHSafeEval via X-Teaming
#
# Usage:
#   ./mh_run.sh
#   DISORDER=depression N_PATIENTS=1 MAX_TURNS=3 ATTACK_MODEL=google/gemini-2.5-flash-lite ./mh_run.sh
#   SKIP_PLAN_GEN=1 ./mh_run.sh    # reuse existing plans

set -euo pipefail

# ── Load .env ─────────────────────────────────────────────────────────────────
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "✅ Loaded .env"
fi

# ── Config ────────────────────────────────────────────────────────────────────
DISORDER="${DISORDER:-depression}"
N_PATIENTS="${N_PATIENTS:-5}"
MAX_TURNS="${MAX_TURNS:-10}"
ATTACK_MODEL="${ATTACK_MODEL:-google/gemini-2.5-flash-lite}"
CONFIG="${CONFIG:-config/mh_config.yaml}"
SKIP_PLAN_GEN="${SKIP_PLAN_GEN:-0}"

echo "════════════════════════════════════════════"
echo " MHSafeEval × X-Teaming"
echo " Disorder    : $DISORDER"
echo " Patients    : $N_PATIENTS"
echo " Max turns   : $MAX_TURNS"
echo " Attack Model: $ATTACK_MODEL"
echo " Config      : $CONFIG"
echo "════════════════════════════════════════════"
echo

# ── Update config on-the-fly ──────────────────────────────────────────────────
python3 - "$CONFIG" "$ATTACK_MODEL" "$MAX_TURNS" "$N_PATIENTS" <<'PYEOF'
import sys, yaml
config_path, attack_model, max_turns, n_patients = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(config_path) as f:
    cfg = yaml.safe_load(f)
cfg["attacker"]["model"]                         = attack_model
cfg["attacker"]["max_turns"]                     = max_turns
cfg["attack_plan_generator"]["num_behaviors"]    = 28 * n_patients
with open(config_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print(f"✅ Config: model={attack_model}, max_turns={max_turns}, num_behaviors={28*n_patients}")
PYEOF

# ── Step 1: Generate behaviors CSV ───────────────────────────────────────────
echo "── Step 1: Generating behaviors CSV ──"
python behaviors/mh_behaviors.py \
    --disorder "$DISORDER" \
    --n-patients "$N_PATIENTS" \
    --out behaviors/mh_behaviors.csv
echo

# ── Step 2: Generate attack plans ────────────────────────────────────────────
if [ "$SKIP_PLAN_GEN" = "1" ]; then
    echo "── Step 2: Skipping plan generation (SKIP_PLAN_GEN=1) ──"
    if [ ! -f strategies/mh_attack_plans.json ]; then
        echo "❌ strategies/mh_attack_plans.json not found!"
        exit 1
    fi
else
    echo "── Step 2: Generating attack plans ──"
    python mh_generate_plans.py --config "$CONFIG"
fi
echo

# ── Step 3: Run attacks ───────────────────────────────────────────────────────
echo "── Step 3: Running attacks ──"
python mh_main_xteam.py --config "$CONFIG"
echo

# ── Step 4: Show metrics ──────────────────────────────────────────────────────
echo "── Step 4: Metrics ──"
LATEST=$(ls -td attacks/*/ 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    TIMESTAMP=$(basename "$LATEST")
    python analytics/mh_metrics.py "$TIMESTAMP" -v \
        --save-json "attacks/${TIMESTAMP}/mh_summary.json"
    echo "✅ Summary saved: attacks/${TIMESTAMP}/mh_summary.json"
else
    echo "⚠️  No attacks/ directory found."
fi