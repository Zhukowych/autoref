#!/usr/bin/env bash
# Start a tmux session with two windows:
#   autoref : venv activated, AUTOREF_CONFIG exported -> run `autoref <cmd>` without -c
#   llm     : local LLM server (vLLM by default)
#
# Usage: scripts/autoref-session.sh -c CONFIG [-s SESSION] [-m MODEL] [-p PORT] [-g GPU]
#                                   [-e VLLM_ENV] [-n] [-k]
#   -c CONFIG    topic config YAML (required unless -k)
#   -s SESSION   tmux session name (default: autoref)
#   -m MODEL     HF model (default: Qwen/Qwen3-14B-AWQ)
#   -p PORT      LLM port (default: 8002)
#   -g GPU       CUDA_VISIBLE_DEVICES (default: 0)
#   -e VLLM_ENV  vLLM venv dir (default: ~/llm/vllm-env)
#   -n           do not start the LLM (only set up the autoref window)
#   -k           kill the session
# Env override: LLM_CMD="custom serve command" replaces the default vllm command.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION=autoref MODEL=Qwen/Qwen3-14B-AWQ PORT=8002 GPU=0 VLLM_ENV="$HOME/llm/vllm-env"
CONFIG="" NO_LLM=0 KILL=0

while getopts "c:s:m:p:g:e:nkh" o; do
  case $o in
    c) CONFIG=$OPTARG ;; s) SESSION=$OPTARG ;; m) MODEL=$OPTARG ;; p) PORT=$OPTARG ;;
    g) GPU=$OPTARG ;; e) VLLM_ENV=$OPTARG ;; n) NO_LLM=1 ;; k) KILL=1 ;;
    *) sed -n '2,/^set -e/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 1 ;;
  esac
done

command -v tmux >/dev/null || { echo "tmux not installed" >&2; exit 1; }
if [ $KILL = 1 ]; then tmux kill-session -t "$SESSION"; exit; fi
[ -n "$CONFIG" ] || { echo "-c CONFIG required" >&2; exit 1; }
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }
CONFIG="$(realpath "$CONFIG")"
[ -x "$ROOT/.venv/bin/autoref" ] || { echo "no $ROOT/.venv; run: python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2; exit 1; }

if [ $NO_LLM = 0 ] && [ -z "${LLM_CMD:-}" ] && [ ! -f "$VLLM_ENV/bin/activate" ]; then
  echo "vLLM venv not found: $VLLM_ENV (pass -e /path/to/venv, or -n to skip the LLM)" >&2; exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' exists; attaching (use -k to kill)"
  exec tmux attach -t "$SESSION"
fi

LLM_CMD="${LLM_CMD:-CUDA_VISIBLE_DEVICES=$GPU vllm serve $MODEL --served-model-name qwen3-14b --host 127.0.0.1 --port $PORT --max-model-len 8192 --max-num-seqs 8 --reasoning-parser qwen3 --gpu-memory-utilization 0.2}"

# window 0: autoref
tmux new-session -d -s "$SESSION" -n autoref -c "$ROOT"
tmux send-keys -t "$SESSION:autoref" \
  "source '$ROOT/.venv/bin/activate' && export AUTOREF_CONFIG='$CONFIG' AUTOREF_LLM_URL='http://127.0.0.1:$PORT/v1' && clear && echo \"AUTOREF_CONFIG=\$AUTOREF_CONFIG\" && echo 'try: autoref llm test | autoref sources test | autoref run'" C-m

# window 1: llm
if [ $NO_LLM = 0 ]; then
  tmux new-window -t "$SESSION:" -n llm -c "$ROOT"
  tmux send-keys -t "$SESSION:llm" "[ -f '$VLLM_ENV/bin/activate' ] && source '$VLLM_ENV/bin/activate'; $LLM_CMD" C-m
fi

tmux select-window -t "$SESSION:autoref"
exec tmux attach -t "$SESSION"
