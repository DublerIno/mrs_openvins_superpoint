#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

export TMUX_SESSION_NAME=ov_tracking
export TMUX_SOCKET_NAME=mrs

DEFAULT_BAG="$SCRIPTPATH/outdoor.mcap"
DEFAULT_CONFIG="$SCRIPTPATH/open_vins/estimator_config.yaml"
DEFAULT_SUPERPOINT_VENV_PYTHON="$HOME/git/SuperPointPretrainedNetwork/venv/bin/python"
DEFAULT_SUPERPOINT_PYTHON="python3"
if [ -x "$DEFAULT_SUPERPOINT_VENV_PYTHON" ]; then
  DEFAULT_SUPERPOINT_PYTHON="$DEFAULT_SUPERPOINT_VENV_PYTHON"
fi

# Priority for each variable: explicit argument > pre-set environment variable > default.
export OV_BAG_PATH="${1:-${OV_BAG_PATH:-$DEFAULT_BAG}}"
export OV_CONFIG_PATH="${2:-${OV_CONFIG_PATH:-$DEFAULT_CONFIG}}"
export OV_SUPERPOINT_PYTHON="${3:-${OV_SUPERPOINT_PYTHON:-$DEFAULT_SUPERPOINT_PYTHON}}"

if [ ! -f "$OV_BAG_PATH" ]; then
  echo "[tmux_tracking] Warning: bag file not found: $OV_BAG_PATH"
fi

if [ ! -f "$OV_CONFIG_PATH" ]; then
  echo "[tmux_tracking] Warning: config file not found: $OV_CONFIG_PATH"
fi

echo "[tmux_tracking] OV_SUPERPOINT_PYTHON=$OV_SUPERPOINT_PYTHON"
echo "[tmux_tracking] Starting with internal defaults/overrides (no manual ros2 arg passing needed)."

tmuxinator start -p ./session_tracking.yml

if [ -z "$TMUX" ]; then
  tmux -L "$TMUX_SOCKET_NAME" a -t "$TMUX_SESSION_NAME"
else
  tmux detach-client -E "tmux -L $TMUX_SOCKET_NAME a -t $TMUX_SESSION_NAME"
fi
