#!/bin/bash
# Download GGUF models from HF Mirror and import into Ollama, then run multi-model experiment
set -e

export OLLAMA_HOST="http://127.0.0.1:11435"
MODEL_DIR="$HOME/.ollama/models/gguf"
mkdir -p "$MODEL_DIR"

# ── Model definitions ──
# Format: name|url|ollama_name
MODELS=(
  "qwen2.5-14b|https://hf-mirror.com/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf|qwen2.5-14b"
  "llama3.1-8b|https://hf-mirror.com/bartowski/Llama-3.1-8B-Instruct-GGUF/resolve/main/Llama-3.1-8B-Instruct-Q4_K_M.gguf|llama3.1-8b"
)

download_model() {
  local name="$1"
  local url="$2"
  local gguf_path="$MODEL_DIR/${name}.gguf"

  if [ -f "$gguf_path" ]; then
    echo "  [$name] Already downloaded ($(du -h "$gguf_path" | cut -f1))"
    return 0
  fi

  echo "  [$name] Downloading from hf-mirror..."
  wget -q --show-progress -O "$gguf_path.part" "$url" 2>&1 && \
    mv "$gguf_path.part" "$gguf_path" && \
    echo "  [$name] Download complete ($(du -h "$gguf_path" | cut -f1))" || \
    { echo "  [$name] Download FAILED"; return 1; }
}

import_to_ollama() {
  local name="$1"
  local gguf_path="$MODEL_DIR/${name}.gguf"
  local ollama_name="$3"

  # Check if already imported
  if curl -s "$OLLAMA_HOST/api/tags" | python3 -c "import json,sys; names=[m['name'] for m in json.load(sys.stdin).get('models',[])]; sys.exit(0 if '$ollama_name' in names else 1)" 2>/dev/null; then
    echo "  [$ollama_name] Already in Ollama"
    return 0
  fi

  echo "  [$ollama_name] Creating Modelfile and importing to Ollama..."
  local modelfile="/tmp/Modelfile_${ollama_name}"
  echo "FROM $gguf_path" > "$modelfile"
  ollama create "$ollama_name" -f "$modelfile" 2>&1 | tail -3
  echo "  [$ollama_name] Import complete"
}

# ── Step 1: Download ──
echo "============================================"
echo "Step 1: Downloading GGUF models"
echo "============================================"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r name url ollama_name <<< "$entry"
  download_model "$name" "$url"
done

# ── Step 2: Import ──
echo ""
echo "============================================"
echo "Step 2: Importing to Ollama"
echo "============================================"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r name url ollama_name <<< "$entry"
  import_to_ollama "$name" "$url" "$ollama_name"
done

echo ""
echo "Models ready:"
curl -s "$OLLAMA_HOST/api/tags" | python3 -c "import json,sys; [print(f'  {m[\"name\"]}') for m in json.load(sys.stdin).get('models',[])]"

# ── Step 3: Run multi-model experiment ──
echo ""
echo "============================================"
echo "Step 3: Running multi-model experiment"
echo "============================================"
cd /home/xxy/vrx_ws/src/marine_env
source /home/xxy/vrx_ws/install/setup.bash 2>/dev/null

python3 run_multimodel.py \
  --scenarios 6,7,9 \
  --repeats 20 \
  --models 7b,14b,llama8b \
  --output multimodel_output \
  --skip-check

echo ""
echo "============================================"
echo "DONE: Multi-model experiment complete"
echo "Results: multimodel_output/"
echo "============================================"
