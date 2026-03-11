#!/bin/bash
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH" || exit 1

export TMUX_SESSION_NAME=superpoint
export TMUX_SOCKET_NAME=sp



#source install/setup.bash

# kill existing session if it exists
tmux -L "$TMUX_SOCKET_NAME" has-session -t "$TMUX_SESSION_NAME" 2>/dev/null
if [ $? -eq 0 ]; then
  tmux -L "$TMUX_SOCKET_NAME" kill-session -t "$TMUX_SESSION_NAME"
fi

tmuxinator start -p ./session.yml

if [ -z "$TMUX" ]; then
  tmux -L "$TMUX_SOCKET_NAME" a -t "$TMUX_SESSION_NAME"
else
  tmux detach-client -E "tmux -L $TMUX_SOCKET_NAME a -t $TMUX_SESSION_NAME"
fi
