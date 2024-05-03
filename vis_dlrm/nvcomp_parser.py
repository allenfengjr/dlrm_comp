import re
from statistics import harmonic_mean

def parse_log(filepath):
    with open(filepath, 'r') as file:
        log_data = file.read()

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
        ratio_match = re.search(r"compressed ratio: ([0-9.]+)", segment)
        comp_thru_match = re.search(r"compression throughput \(GB/s\): ([0-9.]+)", segment)
        decomp_thru_match = re.search(r"decompression throughput \(GB/s\): ([0-9.]+)", segment)
        
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

def print_harmonic_means(harmonic_means):
    for emb in sorted(harmonic_means.keys()):
        cr = harmonic_means[emb]['compression_ratio']
        ct = harmonic_means[emb]['compression_throughput']
        dt = harmonic_means[emb]['decompression_throughput']
        print(f"{cr:.6f} {ct:.6f} {dt:.6f}")



# File path to your log file
file_path = "CMP_ANS_001.log"

# Parse and calculate harmonic means
harmonic_means = parse_log(file_path)
# print(harmonic_means)
print_harmonic_means(harmonic_means)
