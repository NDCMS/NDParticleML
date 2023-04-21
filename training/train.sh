#!/bin/sh

mkdir -p graphs
mkdir -p models
echo Starting...
args="$*"
cmd="python3 ./train.py $args"

$cmd # >> outputs 2>&1