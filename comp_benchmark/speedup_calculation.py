import sys

def calculate_speedup(filepath, bandwidth):
    with open(filepath, 'r') as file:
        data = file.readlines()
    
    results = []
    
    for line in data:
        if line.strip():
            cr, tc, td = map(float, line.strip().split(' '))
            speedup = 1 / (1/cr + bandwidth * ((1/tc) + (1/td)))
            results.append(speedup)
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <file_path> <bandwidth>")
        sys.exit(1)

    file_path = sys.argv[1]
    bandwidth = float(sys.argv[2])
    speedups = calculate_speedup(file_path, bandwidth)
    
    for speedup in speedups:
        print(f"Speed-up: {speedup:.6f}")

'''
Usage python speedup_calculation.py path_logfile.log 0.5 int8

log format: CR Compression_Throughput Decompression_Throughput
'''