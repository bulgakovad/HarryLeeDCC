#!/bin/bash

# Path to your scripts
BG_FIT="python bg_fit.py"
RES_FIT="python bodek_res_fit.py"

# Number of iterations
NUM_ITER=5

for i in $(seq 1 $NUM_ITER); do
    echo "===== Iteration $i ====="
    
    echo "Running bg_fit.py ..."
    $BG_FIT
    if [ $? -ne 0 ]; then
        echo "Error in bg_fit.py, aborting."
        exit 1
    fi

    echo "Running bodek_res_fit.py ..."
    $RES_FIT
    if [ $? -ne 0 ]; then
        echo "Error in bodek_res_fit.py, aborting."
        exit 1
    fi

    echo ""
done

echo "All $NUM_ITER iterations completed."
