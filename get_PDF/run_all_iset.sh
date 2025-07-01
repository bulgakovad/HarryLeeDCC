#!/bin/bash

echo -n "Enter Q2 value: "
read Q2
echo -n "Enter ISET min:"
read ISET_min
echo -n "Enter ISET max:"
read ISET_max

for iset in $(seq $ISET_min $ISET_max); do
    echo "Running ISET = $iset with Q2 = $Q2"
    echo -e "$iset\n$Q2" | ./a.out
done
