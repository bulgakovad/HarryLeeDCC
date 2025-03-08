#!/bin/bash
# Clear the output file before writing new output
> output.txt
# Loop over W values from 1.1 to 3 in steps of 0.1
for w in $(seq 1.1 0.01 3.0); do
  ./exe <<EOF | awk '/E_le/ {print; getline; print}'
1
$w 2.75
10.6
EOF
done >> output.txt
