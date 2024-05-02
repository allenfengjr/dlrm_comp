import re
from statistics import harmonic_mean

def parse_log(log_data):
    # 分段处理每个迭代的数据
    log_segments = log_data.strip().split('Executing: ')
    results = {}

    for segment in log_segments[1:]:  # Skip the initial empty segment
        # 从Completed行提取EMB和iter
        completed_match = re.search(r"Completed: EMB ([0-9]+), Iter ([0-9]+)", segment)
        if not completed_match:
            continue
        
        emb, iter = map(int, completed_match.groups())
        
        # 提取压缩比、压缩吞吐量和解压吞吐量
        ratio_match = re.search(r"compression ratio:\s*([0-9.]+)", segment)
        comp_thru_match = re.search(r"compression e2e throughput:\s*([0-9.]+)\s*GB/s", segment)
        decomp_thru_match = re.search(r"decompression e2e throughput:\s*([0-9.]+)\s*GB/s", segment)
        
        if not (ratio_match and comp_thru_match and decomp_thru_match):
            continue

        ratio = float(ratio_match.group(1))
        comp_thru = float(comp_thru_match.group(1))
        decomp_thru = float(decomp_thru_match.group(1))
        
        if emb not in results:
            results[emb] = {'compression_ratio': [], 'compression_throughput': [], 'decompression_throughput': []}
        
        results[emb]['compression_ratio'].append(ratio)
        results[emb]['compression_throughput'].append(comp_thru)
        results[emb]['decompression_throughput'].append(decomp_thru)
    
    # Calculate harmonic means
    harmonic_means = {}
    for emb, metrics in results.items():
        harmonic_means[emb] = {
            'compression_ratio': harmonic_mean(metrics['compression_ratio']),
            'compression_throughput': harmonic_mean(metrics['compression_throughput']),
            'decompression_throughput': harmonic_mean(metrics['decompression_throughput'])
        }

    return harmonic_means

def read_log_file(filepath):
    with open(filepath, 'r') as file:
        log_data = file.read()
    return log_data

def print_harmonic_means(harmonic_means):
    for emb in sorted(harmonic_means.keys()):
        cr = harmonic_means[emb]['compression_ratio']
        ct = harmonic_means[emb]['compression_throughput']
        dt = harmonic_means[emb]['decompression_throughput']
        print(f"{cr:.6f} {ct:.6f} {dt:.6f}")

# 指定日志文件路径
filepath = "CMP_GPULZ_001.log"

# 从文件读取日志数据
log_data = read_log_file(filepath)
# print(log_data)

# 解析日志数据
results = parse_log(log_data)

# Parse and calculate harmonic means
harmonic_means = parse_log(log_data)
print_harmonic_means(harmonic_means)
