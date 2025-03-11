#!/bin/bash
# Create directory for output .txt data files, if it doesn't exist
mkdir -p output  # If the directory already exists, using -p will not produce an error
# Create directroy for plots, that will be populated by plot.py script
mkdir -p plots 

# Define an array of available Q2 values
Q2_available=(2.99 2.75 2.5 2.25 2.0 1.8 1.6 1.4 1.2 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)

# Loop over Q2 values
for Q2 in "${Q2_available[@]}"; do
    output_file="output/cc_for_Q2=${Q2}_GeV.txt"
    # Clear the output file for this Q2
    > "$output_file"
    
    # Loop over W values from 1.1 to 3.0 in steps of 0.01
    for w in $(seq 1.1 0.01 3.0); do
      ./exe <<EOF | awk '/E_le/ {print; getline; print}'
1
$w $Q2
10.6
EOF
    done >> "$output_file"
done
