#!/bin/sh

mkdir -p logs
mkdir -p graphs
echo Starting...
args="$*"
cmd="python3 ./script_nn.py $args"

$cmd # >> outputs 2>&1