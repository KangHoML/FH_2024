#!/bin/bash

# 공통 경로 설정
DATA_PATH="/home/ho/workspace/datasets/FH_2024/subtask3"

# 현재 날짜와 시간
TIMESTAMP=$(date +"%Y%m%d")

# SEED
SEED=42

# 학습 및 평가 함수 정의
run_task() {
    local task_num=$1
    
    # 학습 실행
    echo "Training for Task $task_num"
    sh run_train.sh --seed $SEED --train_diag_path "${DATA_PATH}/task${task_num}.ddata.wst.txt"
    
    # 평가 실행
    for i in $(seq 1 $task_num); do
        echo "Evaluating on Task $i"
        sh run_eval.sh --seed $SEED --val_diag_path "${DATA_PATH}/cl_eval_task${i}.wst.dev" --ckpt "model_${SEED}_best.pt"
    done
    
    echo "Task $task_num completed"
    echo "------------------------"
}

echo "Starting all tasks at $(date)"

# 각 태스크 실행
for task in {1..6}; do
    run_task $task
done

echo "All tasks completed at $(date)"
