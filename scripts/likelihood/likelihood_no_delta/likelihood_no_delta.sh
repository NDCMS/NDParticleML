#!/bin/sh
printenv
mkdir -p graphs
mkdir -p models
echo Starting...
args="$*"
cmd="python3 ./likelihood_no_delta.py $args"
echo $cmd
$cmd # >> outputs 2>&1
