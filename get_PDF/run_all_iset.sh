#!/bin/bash

echo -n "Enter Q2 value: "
read Q2


for iset in $(seq 400 448); do
    echo "Running ISET = $iset with Q2 = $Q2"
    echo -e "$iset\n$Q2" | ./a.out
done

for iset in $(seq 500 548); do
    echo "Running ISET = $iset with Q2 = $Q2"
    echo -e "$iset\n$Q2" | ./a.out
done
