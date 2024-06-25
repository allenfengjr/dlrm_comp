import os
import subprocess
import numpy as np
from datetime import datetime

def quantization(original_arr, eb, filename):
    eb = (original_arr.max() - original_arr.min()) * eb
    print("absolute error bound, ", eb)
    quantization_arr = np.round(original_arr* (1/(eb*2))).astype(np.int8)
    quantization_arr += abs(quantization_arr.min())+1
    synthetic_outlier = quantization_arr[np.where(quantization_arr>=65536)].astype(np.float32)
    synthetic_outlier.tofile(f"{filename}.outlier")

    print("Range: {}  {}".format(quantization_arr.min(),quantization_arr.max()))
    quantization_arr.tofile(f"{filename}.quan")
    return quantization_arr

# Activate the conda environment
os.system('conda activate dlrm')

# Define directory and filename pattern
DIR = "/N/slate/haofeng/SC_TB_emb/original_padding/"
PATTERN = "EMB_{table}_iter_{iter}.bin.padding"
# QCAT_PATTERN_QUAN = "/home/haofeng/SC_emb_data/SC_Kaggle_emb/quantization/EMB_{table}_iter_{iter}.bin.quan"


# Set environment variables
os.environ['LD_LIBRARY_PATH'] = '/N/u/haofeng/BigRed200/cusz-latest/build:' + os.environ.get('LD_LIBRARY_PATH', '')

# Constants for error bounds
# Terabyte
TIGHTEN_EB_TABLES = {0, 9, 10, 19, 20, 21, 22}
LOOSEN_EB_TABLES = {5, 8, 12, 15, 16, 17, 18, 24, 25}
# Kaggle
# TIGHTEN_EB_TABLES = {2, 3, 9, 11, 15, 20}
# LOOSEN_EB_TABLES = {8, 16, 19, 21, 22, 24}
TIGHTEN_EB_VALUE = 0.01
LOOSEN_EB_VALUE = 0.01
BASE_ERROR_BOUND = 0.01
NUM_TABLES = 26
NUM_ITERATIONS = 23

# Generating EB list with error bounds
EB = []
# decay
for table in range(NUM_TABLES):
    if table in TIGHTEN_EB_TABLES:
        base_eb = TIGHTEN_EB_VALUE
    elif table in LOOSEN_EB_TABLES:
        base_eb = LOOSEN_EB_VALUE
    else:
        base_eb = BASE_ERROR_BOUND
    table_eb = []
    for iter in range(NUM_ITERATIONS):
        # if iter < 24:
        #     eb = (base_eb * 2) - (iter * (base_eb / 24))
        #     # eb = (base_eb * 1)
        # else:
        #     eb = base_eb
        if iter < 12:
            eb = base_eb * 1.8
        else:
            eb = base_eb
        table_eb.append(round(eb, 5))  # Rounding to 5 decimal places
    EB.append(table_eb)

# constant
# for table in range(NUM_TABLES):
#     if table in TIGHTEN_EB_TABLES:
#         base_eb = TIGHTEN_EB_VALUE
#     elif table in LOOSEN_EB_TABLES:
#         base_eb = LOOSEN_EB_VALUE
#     else:
#         base_eb = BASE_ERROR_BOUND

#     table_eb = []
#     for iter in range(NUM_ITERATIONS):
#         if iter < 8:
#             eb = (base_eb * 1.5)
#         else:
#             eb = base_eb
#         table_eb.append(round(eb, 5))  # Rounding to 5 decimal places
#     EB.append(table_eb)

# Update your executable files to include placeholders for dynamic error bounds
EXECUTABLES = {
    #"/N/u/haofeng/BigRed200/cusz-latest/build/cusz": "-z -i {filename} -t f32 -m r2r -e {error_bound} -l 2048x64x128 --report time",
    #"/N/u/haofeng/BigRed200/sz3": "-f -i {filename} -o tempdata.sz.out -2 128 32 -M REL {error_bound} -a",
    # "/home/haofeng/FZ-GPU/fz-gpu": "{filename} 128 32 1 {error_bound}",
    #"/N/u/haofeng/BigRed200/cusz-latest/build/example/bin_hf": "{filename} 2048 64 128 256",
    #"/N/u/haofeng/BigRed200/nvcomp/bin/benchmark_lz4_chunked": "-f {filename}",
    # "/home/haofeng/nvcomp_software/bin/benchmark_deflate_chunked": "-f {filename}",
    "/N/u/haofeng/BigRed200//nvcomp/bin/benchmark_ans_chunked": "-f {filename}",
    #"/N/u/haofeng/BigRed200/ICS23-GPULZ/gpulz": "-i {filename}",
    # "/home/haofeng/qcat-1.3-install/bin/simSZ": "-f 63356 {error_bound} {quan_file} {unpred_file}"
}

# EXECUTABLES = {
#     "/home/haofeng/SZ3/build/bin/sz3": "-f -i {filename} -o tempdata.sz.out -2 131072 16 -M REL 1e-2 -a",
#     "/home/haofeng/sz3/SZ3/build/bin/sz3": "-f -i {filename} -o tempdata.sz.out -2 131072 16 -M REL 1e-2 -a",
#     "/home/haofeng/cusz-latest/build/cusz": "-z -i {filename} -t f32 -m r2r -e 1e-2 -l 131072x16 --report time",
#     "/home/haofeng/FZ-GPU/fz-gpu": "{filename} 131072 16 1 1e-2",
#     "/home/haofeng/nvcomp_software/bin/benchmark_lz4_chunked": "-f {filename}",
#     "/home/haofeng/nvcomp_software/bin/benchmark_deflate_chunked": "-f {filename}",
#     "/home/haofeng/zfp/build/bin/zfp": "-i {filename} -f -2 131072 16 -a 0.01",
#     "/home/haofeng/ICS23-GPULZ/gpulz": "-i {filename}",
# }

LOG_FILES = {
    # Update or maintain log files corresponding to executables as necessary
    "/N/u/haofeng/BigRed200/cusz-latest/build/cusz": "CMP_CHECK_cusz.log",
    "/home/haofeng/sz3_no_pred/bin/sz3": "CMP_NO_PRED.log",
    "/N/u/haofeng/BigRed200/ICS23-GPULZ/gpulz":"CMP_GPULZ.log",
    "/home/haofeng/FZ-GPU/fz-gpu":"CMP_FZGPU.log",
    "/home/haofeng/cusz-latest/build/example/bin_hf":"CMP_Huffman.log",
    "/N/u/haofeng/BigRed200/nvcomp/bin/benchmark_lz4_chunked": "CMP_nvCOMP_LZ4.log",
    "/home/haofeng/nvcomp_software/bin/benchmark_deflate_chunked": "CMP_nvCOMP_deflate.log",
    "/home/haofeng/qcat-1.3-install/bin/simSZ": "CMP_simSZ.log",
    "/N/u/haofeng/BigRed200//nvcomp/bin/benchmark_ans_chunked": "CMP_ANS.log"
}

# Change loop order to compressor-EMB-iter
for exec_path, exec_args in EXECUTABLES.items():
    for table in range(len(EB)):  # Loop through tables based on the EB list
        for iter in range(1,len(EB[table])):  # Loop through iterations for each table
            # Construct the filename
            filename = PATTERN.format(table=table, iter=iter)
            filename = os.path.join(DIR, filename)
            # original_data = np.fromfile(filename,dtype=np.float32)
            # value_range = original_data.max() -original_data.min()
            error_bound = EB[table][iter]
            if os.path.isfile(filename):
                args = exec_args.format(filename=filename, error_bound=error_bound)
                log_file_name = LOG_FILES.get(exec_path, "default.log")

                with open(log_file_name, 'a') as log_file:
                    log_file.write(f"Executing: {exec_path} {args}\n")
                    result = subprocess.run([exec_path] + args.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    log_file.write(result.stdout.decode())
                    log_file.write(f"Completed: EMB {table}, Iter {iter}\n")
                    # Optionally gzip the output
                    # subprocess.run(["gzip", "-v", filename])
            else:
                general_log_file = "general_errors.log"
                with open(general_log_file, 'a') as log_file:
                    log_file.write(f"File {filename} does not exist\n")

