#!/bin/bash

CONFIG=$1
NUM_WORKERS=$2
NUM_GPUS=$3
STEP=${4:-1}
PORT=${5:-29500}

print_banner() {
    echo "=================================="
    echo "Executing: $1"
    echo "=================================="
}

run_command() {
    if [ $STEP -le $1 ]; then
        print_banner "$2"
        eval "$2"
        if [ $? -ne 0 ]; then
            exit 1
        fi
    fi
}

command_list=(

    "python -m nuplan_scripts.stack_RGB_point_cloud --config $CONFIG --num_workers $NUM_WORKERS"
)

for i in $(seq 1 ${#command_list[@]}); do
    run_command $i "${command_list[$i - 1]}"
done
