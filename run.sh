#!/bin/bash
#set the value of Q2
Q2=2.75
# Clear the output file before writing new output
> cc_for_Q2=${Q2}_GeV.txt
# Loop over W values from 1.1 to 3.0 in steps of 0.01
for w in $(seq 1.1 0.01 3.0); do
  ./exe <<EOF | awk '/E_le/ {print; getline; print}'
1
$w $Q2
10.6
EOF
done >> cc_for_Q2=${Q2}_GeV.txt