#!/bin/bash

# 공통 경로 설정
DATA_PATH="../../Dataset/subtask3"
LOG_DIR="./logs"

# 로그 디렉토리 생성
mkdir -p $LOG_DIR

# 현재 날짜와 시간
TIMESTAMP=$(date +"%Y%m%d")

# 학습 및 평가 함수 정의
run_task() {
    local task_num=$1
    local log_file="${LOG_DIR}/task${task_num}_${TIMESTAMP}.log"
    
    echo "Running Task $task_num" | tee -a "$log_file"
    
    # 학습 실행
    echo "Training for Task $task_num" | tee -a "$log_file"
    sh run_train.sh --seed 42 --train_diag_path "${DATA_PATH}/task${task_num}.ddata.wst.txt" >> "$log_file" 2>&1
    
    # 평가 실행
    for i in $(seq 1 $task_num); do
        echo "Evaluating on Task $i" | tee -a "$log_file"
        sh run_eval.sh --seed 42 --val_diag_path "${DATA_PATH}/cl_eval_task${i}.wst.dev" >> "$log_file" 2>&1
    done
    
    echo "Task $task_num completed" | tee -a "$log_file"
    echo "------------------------" | tee -a "$log_file"
}

# 전체 실행 로그 파일
MAIN_LOG="${LOG_DIR}/all_tasks_${TIMESTAMP}.log"

echo "Starting all tasks at $(date)" | tee "$MAIN_LOG"

# 각 태스크 실행
for task in {1..6}; do
    run_task $task 2>&1 | tee -a "$MAIN_LOG"
done

echo "All tasks completed at $(date)" | tee -a "$MAIN_LOG"
