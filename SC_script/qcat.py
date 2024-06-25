import os
import subprocess
import numpy as np
# Activate the conda environment
os.system('conda activate dlrm')

# Define directories and filename patterns
QCAT_DIR = "/home/haofeng/SC_emb_data/SC_TB_emb/qcat"
PATTERN = "/home/haofeng/SC_emb_data/SC_TB_emb/EMB_{table}_iter_{iter}.bin"
QCAT_PATTERN_QUAN = "/home/haofeng/SC_emb_data/SC_TB_emb/quantization/EMB_{table}_iter_{iter}.bin.quan"
QCAT_PATTERN_OUTLIER = "/home/haofeng/SC_emb_data/SC_TB_emb/outlier/EMB_{table}_iter_{iter}.bin.outlier"

# Set environment variables
os.environ['LD_LIBRARY_PATH'] = '/home/haofeng/anaconda3/envs/dlrm/lib:/home/haofeng/cusz-latest/build:' + os.environ.get('LD_LIBRARY_PATH', '')

# Constants for error bounds
TIGHTEN_EB_TABLES = {0, 9, 10, 19, 20, 21, 22}
LOOSEN_EB_TABLES = {5, 8, 12, 15, 16, 17, 18, 24, 25}
TIGHTEN_EB_VALUE = 0.01
LOOSEN_EB_VALUE = 0.05
BASE_ERROR_BOUND = 0.03
NUM_TABLES = 26
NUM_ITERATIONS = 24

# Generating EB list with error bounds
EB = []
for table in range(NUM_TABLES):
    if table in TIGHTEN_EB_TABLES:
        base_eb = TIGHTEN_EB_VALUE
    elif table in LOOSEN_EB_TABLES:
        base_eb = LOOSEN_EB_VALUE
    else:
        base_eb = BASE_ERROR_BOUND

    table_eb = []
    for iter in range(NUM_ITERATIONS):
        if iter < 8:
            eb = (base_eb )
        else:
            eb = base_eb
        table_eb.append(round(eb, 5))  # Rounding to 5 decimal places
    EB.append(table_eb)

# Update your executable files to handle special commands
EXECUTABLES = {
    "/home/haofeng/qcat-1.3-install/bin/simSZ": "-f 63356 {error_bound} {quan_file} {unpred_file}"
    # Add other executables as needed
}

# Define log files for executables
LOG_FILES = {
    "/home/haofeng/qcat-1.3-install/bin/simSZ": "CMP_simSZ.log"
    # Add other log file paths as needed
}

# Processing loop
for exec_path, exec_args in EXECUTABLES.items():
    for table in range(NUM_TABLES):
        for iter in range(1,NUM_ITERATIONS):
            error_bound = EB[table][iter]
            quan_file = QCAT_PATTERN_QUAN.format(table=table, iter=iter)
            outlier_file = QCAT_PATTERN_OUTLIER.format(table=table, iter=iter)
            filename = PATTERN.format(table=table, iter=iter)
            original_data = np.fromfile(filename,dtype=np.float32)
            value_range = original_data.max() -original_data.min()
            error_bound *= value_range

            if os.path.isfile(quan_file) and os.path.isfile(outlier_file):
                args = exec_args.format(error_bound=error_bound, quan_file=quan_file, unpred_file=outlier_file)
                log_file_name = LOG_FILES.get(exec_path, "default.log")

                with open(log_file_name, 'a') as log_file:
                    log_file.write(f"Executing: {exec_path} {args}\n")
                    log_file.write(f"Value Range{value_range}\n")
                    result = subprocess.run([exec_path] + args.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    log_file.write(result.stdout.decode())
                    log_file.write(f"Completed: EMB {table}, Iter {iter}\n")
            else:
                general_log_file = "general_errors.log"
                with open(general_log_file, 'a') as log_file:
                    log_file.write(f"Files {quan_file} and/or {outlier_file} do not exist\n")
