#!/bin/bash

# Unified Rainbow Teaming - Each model tests itself

# Models to evaluate (each model tests itself)
MODELS=(
    "anthropic/claude-haiku-4.5"
    # "minimax/minimax-m2.5"
    # "xiaomi/mimo-v2-flash"
    # "openai/gpt-3.5-turbo"
    # "meta-llama/llama-3.1-8b-instruct"
    # "google/gemini-2.5-flash"
    # "deepseek/deepseek-v3.2"
    # "google/gemma-4-26b-a4b-it"
)

DISORDERS=("depression" "delusion" "psychosis")

# 🔧 Disorder별 실제 환자 수 (하드코딩)
declare -A PATIENT_COUNTS
PATIENT_COUNTS["depression"]=51
PATIENT_COUNTS["delusion"]=4
PATIENT_COUNTS["psychosis"]=3

MAX_TURNS=10
NUM_ITERATIONS=1
MAX_RETRIES=3  # Max retries per turn (1=no retry, 5=retry up to 5 times)

echo "🌈 Unified Rainbow Teaming - Self-Testing Evaluation"
echo "=================================================================="
echo ""
echo "Configuration:"
echo "  - Models (each tests itself): ${#MODELS[@]} models"
for MODEL in "${MODELS[@]}"; do
    echo "    • $MODEL"
done
echo "  - Disorders & Patient Counts:"
for DISORDER in "${DISORDERS[@]}"; do
    echo "    • $DISORDER: ${PATIENT_COUNTS[$DISORDER]} patients"
done
echo "  - Max Turns per Patient: $MAX_TURNS"
echo "  - Number of Iterations: $NUM_ITERATIONS"
echo "  - Max Retries per Turn: $MAX_RETRIES"
echo "  - Evaluation: 5-point severity scale"
echo "  - Strategy: Always Rainbow + Adaptive Learning"
echo ""
echo "=================================================================="

TOTAL_MODELS=${#MODELS[@]}
MODEL_COUNTER=0

for MODEL in "${MODELS[@]}"; do
    MODEL_COUNTER=$((MODEL_COUNTER + 1))
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  MODEL $MODEL_COUNTER / $TOTAL_MODELS: $MODEL (self-testing)"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    MODEL_START_TIME=$(date +%s)
    
    for ITERATION in $(seq 1 $NUM_ITERATIONS); do
        echo ""
        echo "┌────────────────────────────────────────────────┐"
        echo "│  ITERATION $ITERATION / $NUM_ITERATIONS for $MODEL"
        echo "└────────────────────────────────────────────────┘"
        echo ""
        
        for DISORDER in "${DISORDERS[@]}"; do
            # 🔧 Disorder별 환자 수 가져오기
            NUM_PATIENTS=${PATIENT_COUNTS[$DISORDER]}
            
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "Model: $MODEL | Disorder: $DISORDER | Patients: $NUM_PATIENTS | Iteration: $ITERATION"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            
            START_TIME=$(date +%s)
            
            python main.py \
                --disorder_type "$DISORDER" \
                --model "$MODEL" \
                --max_turns "$MAX_TURNS" \
                --num_patients "$NUM_PATIENTS" \
                --mutation_strategy adaptive \
                --learning_rate 1.0 \
                --iteration "$ITERATION" \
                --max_retries_per_turn "$MAX_RETRIES"
            
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            
            echo ""
            echo "✓ $DISORDER complete with $NUM_PATIENTS patients (took ${ELAPSED}s)"
            echo ""
        done
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 Aggregating results for $MODEL - Iteration $ITERATION..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Run aggregation
        python -c "
import asyncio
from main import aggregate_iteration_results
aggregate_iteration_results($ITERATION, '$MODEL')
"
        
        echo ""
        echo "✓ Iteration $ITERATION / $NUM_ITERATIONS complete for $MODEL!"
        echo ""
    done
    
    MODEL_END_TIME=$(date +%s)
    MODEL_ELAPSED=$((MODEL_END_TIME - MODEL_START_TIME))
    MODEL_ELAPSED_MIN=$((MODEL_ELAPSED / 60))
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  ✓ MODEL $MODEL_COUNTER / $TOTAL_MODELS COMPLETE: $MODEL"
    echo "║  Total time: ${MODEL_ELAPSED_MIN} minutes (${MODEL_ELAPSED}s)"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Brief pause between models
    sleep 2
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 All $TOTAL_MODELS models × $NUM_ITERATIONS iterations complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Configuration Summary:"
echo "  - Each model tested itself (counselor + agents)"
echo "  - Disorder-specific patient counts:"
for DISORDER in "${DISORDERS[@]}"; do
    echo "    • $DISORDER: ${PATIENT_COUNTS[$DISORDER]} patients"
done
echo "  - Max Retries: $MAX_RETRIES"
echo ""
echo "Results saved to:"
echo "  - eval_outputs_unified/{model}/successful_attacks_iter{N}.txt"
echo "  - eval_outputs_unified/{model}/evaluation_summary_iter{N}.json"
echo "  - eval_outputs_unified/{model}/unified_archive_{disorder}.json"
echo "  - eval_outputs_unified/{model}/unified_strategies_{disorder}.json"
echo ""
echo "Models evaluated:"
for MODEL in "${MODELS[@]}"; do
    echo "  ✓ $MODEL"
done
echo ""