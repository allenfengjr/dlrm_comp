import re
from statistics import harmonic_mean

def parse_log(filepath):
    with open(filepath, 'r') as file:
        log_data = file.read()

    # Process each iteration's data separately
    log_segments = log_data.strip().split('Executing: ')
    results = {}

    for segment in log_segments[1:]:  # Skip the initial empty segment
        # Extract EMB and Iter from the Completed line
        completed_match = re.search(r"Completed: EMB ([0-9]+), Iter ([0-9]+)", segment)
        if not completed_match:
            continue
        
        emb, iter = map(int, completed_match.groups())
        
        # Extract compression ratio, compression throughput, and decompression throughput
        ratio_match = re.search(r"Huffman CR = sizeof\(E\) \* len / outlen\", where outlen is byte count:\s*([\d.]+)", segment)
        comp_thru_match = re.search(r"\[psz::info::res::comp_hf_encode\] shortest time \(ms\): .*highest throughput \(GiB/s\): ([\d.]+)", segment)
        decomp_thru_match = re.search(r"\[psz::info::res::decomp_hf_decode\] shortest time \(ms\): .*highest throughput \(GiB/s\): ([\d.]+)", segment)
        
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
    print("Compression Ratios (CR):")
    for emb, metrics in sorted(harmonic_means.items()):
        cr = metrics['compression_ratio']
        print(f"CR, {cr:.6f}")

    print("\nCompression Throughputs:")
    for emb, metrics in sorted(harmonic_means.items()):
        ct = metrics['compression_throughput']
        print(f"Comp, {ct:.6f}")

    print("\nDecompression Throughputs:")
    for emb, metrics in sorted(harmonic_means.items()):
        dt = metrics['decompression_throughput']
        print(f"Decomp, {dt:.6f}")


# Example file path to your log file
file_path = "../vis_dlrm/CMP_Huffman.log"

# Parse and calculate harmonic means
harmonic_means = parse_log(file_path)
print_harmonic_means(harmonic_means)
