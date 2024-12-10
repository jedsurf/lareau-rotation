#!/bin/bash

# SHELL="/bin/bash"
# ctrl+b, d to detach session 
# tmux new-session -s mysession # (starts session)
# tmux kill-session -t mysession # (kills session)
# conda activate $1
jupyter notebook --no-browser --port=8889
