#!/bin/bash

PROJECTDIR="/home/jcc//Documents/entanglement/mps-gme"

source ~/Documents/entanglement/.env/bin/activate
datetime=$(date '+%Y-%m-%d-%H%M')
python3 $PROJECTDIR/bin/main.py $datetime $PROJECTDIR
