#!/bin/sh

mkdir -p graphs
mkdir -p models
echo Starting...
args="$*"
cmd="python3 ./likelihood.py $args"

$cmd # >> outputs 2>&1
