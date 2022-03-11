#! /bin/bash

a=( "a"
    "b"
    "c"
    "r"
    "g")
echo ${a[*]}

for o in "${a[@]}"; do
    echo $o
done
