#!/bin/sh

mkdir -p graphs
echo Starting...
cmd="python3 ./profile_cQei.py"

echo $cmd # >> outputs 2>&1
$cmd