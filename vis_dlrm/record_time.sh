#!/bin/bash

# Name of the executable
EXECUTABLE="/home/haofeng/zfp/build/bin/zfp -x cuda -i /home/haofeng/TB_emb_8m/throughput_test.bin -f -2 131072 16 -r 8 -s"

# Log file to save elapsed time
LOGFILE="execution_times.log"

# Get the start time in microseconds
START_TIME=$(date "+%s%6N")

# Execute the program
$EXECUTABLE "$@"

# Get the end time in microseconds
END_TIME=$(date "+%s%6N")

# Calculate elapsed time in microseconds
ELAPSED_TIME=$((END_TIME - START_TIME))

# Log the elapsed time
echo "${ELAPSED_TIME} microseconds" >> $LOGFILE

