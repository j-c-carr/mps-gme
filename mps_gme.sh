#!/bin/bash

PROJECTDIR="/home/jcc//Documents/entanglement/mps-gme"

source ~/Documents/entanglement/.env/bin/activate
datetime=$(date '+%Y-%m-%d-%H%M')

if [[ $1 == "debug" ]]; then
    python3 -m pudb $PROJECTDIR/bin/main.py $datetime $PROJECTDIR
else 
    python3 $PROJECTDIR/bin/main.py $datetime $PROJECTDIR
fi
