#!/bin/bash
# Check training status on RunPod
#
# Usage:
#   ./scripts/check_training_status.sh

RUNPOD_HOST="root@193.183.22.54"
RUNPOD_PORT="1795"

echo "========================================"
echo "  RUNPOD TRAINING STATUS"
echo "========================================"
echo "  Time: $(date)"
echo ""

# Check running processes
echo "Running processes:"
ssh -p $RUNPOD_PORT $RUNPOD_HOST "ps aux | grep train_agent | grep -v grep" 2>/dev/null

RUNNING=$(ssh -p $RUNPOD_PORT $RUNPOD_HOST "ps aux | grep train_agent | grep -v grep | wc -l" 2>/dev/null)

echo ""
echo "----------------------------------------"
if [ "$RUNNING" -eq 0 ]; then
    echo "STATUS: TRAINING COMPLETE"
    echo ""
    echo "Models ready for download:"
    ssh -p $RUNPOD_PORT $RUNPOD_HOST "ls -la /workspace/models/" 2>/dev/null
    echo ""
    echo "Run: ./scripts/download_models_runpod.sh"
else
    echo "STATUS: $RUNNING AGENTS TRAINING"
    echo ""
    echo "Latest log entries:"
    for i in 0 1 2; do
        echo ""
        echo "--- agent_00$i ---"
        ssh -p $RUNPOD_PORT $RUNPOD_HOST "tail -3 /workspace/logs/agent_00$i.log" 2>/dev/null
    done
fi
echo "========================================"
