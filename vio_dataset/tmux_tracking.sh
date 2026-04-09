#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

export TMUX_SESSION_NAME=ov_tracking
export TMUX_SOCKET_NAME=mrs

DEFAULT_BAG="$SCRIPTPATH/outdoor.mcap"
DEFAULT_CONFIG="$SCRIPTPATH/open_vins/estimator_config.yaml"

export OV_BAG_PATH="${1:-$DEFAULT_BAG}"
export OV_CONFIG_PATH="${2:-$DEFAULT_CONFIG}"

if [ ! -f "$OV_BAG_PATH" ]; then
  echo "[tmux_tracking] Warning: bag file not found: $OV_BAG_PATH"
fi

if [ ! -f "$OV_CONFIG_PATH" ]; then
  echo "[tmux_tracking] Warning: config file not found: $OV_CONFIG_PATH"
fi

tmuxinator start -p ./session_tracking.yml

if [ -z "$TMUX" ]; then
  tmux -L "$TMUX_SOCKET_NAME" a -t "$TMUX_SESSION_NAME"
else
  tmux detach-client -E "tmux -L $TMUX_SOCKET_NAME a -t $TMUX_SESSION_NAME"
fi
