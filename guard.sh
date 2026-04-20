#!/bin/bash
# 训练守护脚本：训练崩溃或完成后自动关机
# 用法: bash guard.sh <训练进程PID>

TRAINING_PID=$1
LOG_FILE="experiments_grpo/grpo_from_sft/run_grpo_resume.log"
GUARD_LOG="experiments_grpo/grpo_from_sft/guard.log"

if [ -z "$TRAINING_PID" ]; then
    echo "用法: bash guard.sh <PID>"
    exit 1
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$GUARD_LOG"
}

log "守护启动，监控 PID=$TRAINING_PID"

# 等待训练进程结束
wait $TRAINING_PID
EXIT_CODE=$?

log "训练进程已退出，exit code=$EXIT_CODE"

# 检查是否是 OOM 或其他错误
if grep -q "OutOfMemoryError\|CUDA out of memory\|Traceback\|Error" "$LOG_FILE" 2>/dev/null | tail -20; then
    log "检测到错误（OOM 或异常），准备关机"
else
    if [ $EXIT_CODE -eq 0 ]; then
        log "训练正常完成，准备关机"
    else
        log "训练异常退出 (exit=$EXIT_CODE)，准备关机"
    fi
fi

log "60 秒后关机，如需取消请执行: kill $$ 或 shutdown -c"
sleep 60
log "执行关机..."
shutdown -h now
