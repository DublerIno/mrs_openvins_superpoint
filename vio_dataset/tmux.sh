#!/bin/bash

# Absolute path to this script. /home/user/bin/foo.sh
SCRIPT=$(readlink -f $0)
# Absolute path this script is in. /home/user/bin
SCRIPTPATH=`dirname $SCRIPT`
cd "$SCRIPTPATH"

export TMUX_SESSION_NAME=docker
export TMUX_SOCKET_NAME=mrs

DEFAULT_CONFIG="$SCRIPTPATH/open_vins/estimator_config.yaml"
DEFAULT_SUPERPOINT_VENV_PYTHON="$HOME/git/SuperPointPretrainedNetwork/venv/bin/python"
DEFAULT_SUPERPOINT_PYTHON="python3"
if [ -x "$DEFAULT_SUPERPOINT_VENV_PYTHON" ]; then
  DEFAULT_SUPERPOINT_PYTHON="$DEFAULT_SUPERPOINT_VENV_PYTHON"
fi

# Priority: arg > existing env > default
export OV_CONFIG_PATH="${1:-${OV_CONFIG_PATH:-$DEFAULT_CONFIG}}"
export OV_SUPERPOINT_PYTHON="${2:-${OV_SUPERPOINT_PYTHON:-$DEFAULT_SUPERPOINT_PYTHON}}"

echo "[tmux] OV_CONFIG_PATH=$OV_CONFIG_PATH"
echo "[tmux] OV_SUPERPOINT_PYTHON=$OV_SUPERPOINT_PYTHON"

# start tmuxinator
tmuxinator start -p ./session.yml

# if we are not in tmux
if [ -z $TMUX ]; then

  # just attach to the session
  tmux -L $TMUX_SOCKET_NAME a -t $TMUX_SESSION_NAME

# if we are in tmux
else

  # switch to the newly-started session
  tmux detach-client -E "tmux -L $TMUX_SOCKET_NAME a -t $TMUX_SESSION_NAME"

fi
