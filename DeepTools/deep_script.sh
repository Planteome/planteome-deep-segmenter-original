#!/bin/bash

args=("$@")
echo Number of arguments passed: $#

file="./DeepTools/deep_config.txt"
while IFS= read -r varname; do
	printf '%s\n' "$varname"
done < "$file"
read -r value<$file

source $value # activate proper environment in deep_config

python ./DeepTools/deep_python.py "$@" > results.txt
