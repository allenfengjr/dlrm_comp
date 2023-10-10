#!/bin/bash

# Define directory and filename pattern
conda activate dlrm
DIR="/home/haofeng/TB_emb_8m"
PATTERN="embedding_output_vector_table_{table}_epoch_{epoch}_iter_{iter}.bin"
export LD_LIBRARY_PATH=/home/haofeng/anaconda3/envs/dlrm/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/haofeng/cusz-latest/build:$LD_LIBRARY_PATH

# Define your executable files and their arguments
# Use $filename as a placeholder in the arguments
declare -A EXECUTABLES=(
    #["/home/haofeng/SZ3/build/bin/sz3"]="-f -i %FILENAME% -o tempdata.sz.out -2 131072 16 -M REL 1e-2 -a"
    #["/home/haofeng/cusz-latest/build/cusz"]="-z -i %FILENAME% -t f32 -m r2r -e 1e-2 -l 131072x16 --report time"
    #["/home/haofeng/FZ-GPU/fz-gpu"]="%FILENAME% 131072 16 1 1e-2"
    #["/home/haofeng/nvcomp_software/bin/benchmark_lz4_chunked"]="-f %FILENAME%"
    #["/home/haofeng/nvcomp_software/bin/benchmark_deflate_chunked"]="-f %FILENAME%"
    ["/home/haofeng/zfp/build/bin/zfp"]="-i %FILENAME% -f -2 131072 16 -a 0.001"
)

# Redirecting output to a file
LOG_FILE="zfp_script_output.log"
echo "Script started on $(date)" > $LOG_FILE

# Loop over tables, epochs, iters (you can define ranges as per your requirement)
for table in {0..25}; do
    for epoch in {10..10}; do
        for iter in {0..0}; do
            # Construct the filename
            filename="${DIR}/${PATTERN/\{table\}/$table}"
            filename="${filename/\{epoch\}/$epoch}"
            filename="${filename/\{iter\}/$iter}"
            
            quant_filename="${filename}.quan"
            # Check if file exists
            if [[ -f $filename ]]; then
                # Loop through each executable and its arguments
                for exec in "${!EXECUTABLES[@]}"; do
                    # Replace $filename placeholder with the actual filename
                    ARGS=${EXECUTABLES[$exec]//%FILENAME%/$filename}
                    
                    # Log the command being executed
                    echo "Executing: $exec $ARGS" >> $LOG_FILE
                    
                    # Call the executable and append output to the log
                    $exec $ARGS >> $LOG_FILE 2>&1

                    echo "Completed: Table $table" >> $LOG_FILE
                done
                #gzip -v $quant_filename
            else
                echo "File $filename does not exist" >> $LOG_FILE
            fi
        done
    done
done

echo "Script ended on $(date)" >> $LOG_FILE
