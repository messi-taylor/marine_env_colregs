#!/bin/bash
# Monitor downloads, import into Ollama, then run multi-model experiment
set -e

export OLLAMA_HOST="http://127.0.0.1:11435"
MODEL_DIR="$HOME/.ollama/models/gguf"
QWEN_FILE="$MODEL_DIR/qwen2.5-14b-q4_k_m.gguf"
LLAMA_FILE="$MODEL_DIR/llama3.1-8b-q4_k_m.gguf"

echo "Waiting for downloads to complete..."

# Wait for qwen download (9GB, check if wget still running)
while ps -p 77557 -o pid= > /dev/null 2>&1; do
  SIZE=$(stat -c%s "$QWEN_FILE" 2>/dev/null | awk '{printf "%.1f GB", $1/1073741824}')
  echo "  qwen2.5:14b — $SIZE (downloading...)"
  sleep 30
done
echo "  qwen2.5:14b — download complete ($(du -h "$QWEN_FILE" | cut -f1))"

# Wait for llama download (~5GB)
while ps -p 77608 -o pid= > /dev/null 2>&1; do
  SIZE=$(stat -c%s "$LLAMA_FILE" 2>/dev/null | awk '{printf "%.1f GB", $1/1073741824}')
  echo "  llama3.1:8b — $SIZE (downloading...)"
  sleep 30
done
echo "  llama3.1:8b — download complete ($(du -h "$LLAMA_FILE" | cut -f1))"

# Import into Ollama
echo ""
echo "Importing models into Ollama..."

echo "FROM $QWEN_FILE" > /tmp/Modelfile_qwen14b
ollama create qwen2.5:14b -f /tmp/Modelfile_qwen14b 2>&1 | tail -2
echo "  qwen2.5:14b — imported"

echo "FROM $LLAMA_FILE" > /tmp/Modelfile_llama8b
ollama create llama3.1:8b -f /tmp/Modelfile_llama8b 2>&1 | tail -2
echo "  llama3.1:8b — imported"

# Run experiment
echo ""
echo "Running multi-model experiment on S06/S07/S09..."
cd /home/xxy/vrx_ws/src/marine_env
source /home/xxy/vrx_ws/install/setup.bash 2>/dev/null

python3 run_multimodel.py \
  --scenarios 6,7,9 \
  --repeats 20 \
  --models 7b,14b,llama8b \
  --output multimodel_output \
  --skip-check

echo ""
echo "DONE"
