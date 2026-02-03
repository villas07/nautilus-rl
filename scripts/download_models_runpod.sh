#!/bin/bash
# Download trained models from RunPod
#
# Usage:
#   ./scripts/download_models_runpod.sh [batch_name]
#
# Example:
#   ./scripts/download_models_runpod.sh batch2

# Configuration
RUNPOD_HOST="root@193.183.22.54"
RUNPOD_PORT="1795"
REMOTE_MODELS="/workspace/models"
REMOTE_LOGS="/workspace/logs"

# Batch name (default: batch2)
BATCH_NAME="${1:-batch2}"
LOCAL_DIR="./models_${BATCH_NAME}"
LOCAL_LOGS="./logs_${BATCH_NAME}"

echo "========================================"
echo "  DOWNLOAD MODELS FROM RUNPOD"
echo "========================================"
echo "  Host: $RUNPOD_HOST"
echo "  Port: $RUNPOD_PORT"
echo "  Remote models: $REMOTE_MODELS"
echo "  Local dir: $LOCAL_DIR"
echo ""

# Check if training is still running
echo "Checking if training is complete..."
RUNNING=$(ssh -p $RUNPOD_PORT $RUNPOD_HOST "ps aux | grep train_agent | grep -v grep | wc -l")

if [ "$RUNNING" -gt 0 ]; then
    echo "WARNING: $RUNNING training processes still running!"
    echo ""
    ssh -p $RUNPOD_PORT $RUNPOD_HOST "ps aux | grep train_agent | grep -v grep"
    echo ""
    read -p "Download anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Create local directories
mkdir -p "$LOCAL_DIR"
mkdir -p "$LOCAL_LOGS"

# List available models
echo ""
echo "Available models on RunPod:"
ssh -p $RUNPOD_PORT $RUNPOD_HOST "ls -la $REMOTE_MODELS/"

# Download models
echo ""
echo "Downloading models..."
scp -P $RUNPOD_PORT -r "$RUNPOD_HOST:$REMOTE_MODELS/"* "$LOCAL_DIR/"

# Download logs
echo ""
echo "Downloading logs..."
scp -P $RUNPOD_PORT -r "$RUNPOD_HOST:$REMOTE_LOGS/"* "$LOCAL_LOGS/"

# Verify download
echo ""
echo "Downloaded models:"
ls -la "$LOCAL_DIR/"

echo ""
echo "========================================"
echo "  DOWNLOAD COMPLETE"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Run validation:"
echo "     python scripts/validate_batch.py --models-dir $LOCAL_DIR"
echo ""
echo "  2. Or run individual filter:"
echo "     python validation/run_validation.py --models-dir $LOCAL_DIR --filter basic"
echo ""
