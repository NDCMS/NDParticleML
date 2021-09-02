#!/bin/sh

mkdir -p graphs
mkdir -p models
echo Starting...
args="$*"
cmd="python3 ./likelihood.py $args"
echo $cmd
$cmd # >> outputs 2>&1
