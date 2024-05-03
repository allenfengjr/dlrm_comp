import sys

def calculate_speedup_and_times(filepath, bandwidth, data_size):
    with open(filepath, 'r') as file:
        data = file.readlines()
    
    results = []
    
    for line in data:
        if line.strip():
            cr, tc, td = map(float, line.strip().split(' '))
            # Calculate speed-up
            speedup = 1 / (1/cr + bandwidth * ((1/tc) + (1/td)))
            
            # Calculate compression and decompression time in seconds
            compression_time = data_size / tc  # Given tc is in GB/s
            decompression_time = data_size / td  # Given td is in GB/s
            
            # Calculate time to transmit compressed data
            compressed_data_size = data_size / cr
            transmission_time_compressed = compressed_data_size / bandwidth  # Bandwidth in GB/s
            
            # Calculate time to transmit original data
            transmission_time_original = data_size / bandwidth
            
            results.append({
                'speedup': speedup,
                'compression_time': compression_time,
                'decompression_time': decompression_time,
                'transmission_time_compressed': transmission_time_compressed,
                'transmission_time_original': transmission_time_original
            })
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <file_path> <bandwidth>")
        sys.exit(1)

    file_path = sys.argv[1]
    bandwidth = float(sys.argv[2])
    data_size = 1  # Assume data size is 1 GB for this example
    results = calculate_speedup_and_times(file_path, bandwidth, data_size)
    
    # for result in results:
    #     print(f"Speed-up: {result['speedup']:.6f}")
    #     print(f"Compression time: {result['compression_time'] * 1000:.6f} ms")
    #     print(f"Decompression time: {result['decompression_time'] * 1000:.6f} ms")
    #     print(f"Time to transmit compressed data: {result['transmission_time_compressed'] * 1000:.6f} ms")
    #     print(f"Time to transmit original data: {result['transmission_time_original'] * 1000:.6f} ms")

    for result in results:
        print(f"{result['compression_time'] * 1000:.6f}, {result['decompression_time'] * 1000:.6f}, {result['transmission_time_compressed'] * 1000:.6f}, {result['transmission_time_original'] * 1000:.6f}")
