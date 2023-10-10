import re

def read_log_file(filename, n):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
        return lines[::n]  # Return every nth line
    except FileNotFoundError:
        return None

def extract_compression_ratio(line):
    # Check if the line contains "compressed ratio" information
    if "compressed ratio" in line:
        # Use string manipulation to extract the ratio part
        parts = line.split()
        ratio_index = parts.index("ratio:")
        if ratio_index < len(parts) - 1:
            ratio_value = parts[ratio_index + 1]
            return ratio_value
    return None

'''
def extract_compression_ratio(line):
    # 使用正则表达式匹配压缩比率行
    match = re.search(r'compression ratio: (\d+\.\d+)', line)
    if match:
        ratio = match.group(1)
        return ratio
    return None

def extract_compression_ratio(log_text):
    # Use regular expressions to find the compression ratio value
    compression_ratio_pattern = r'compression ratio\s+([\d.]+)'
    match = re.search(compression_ratio_pattern, log_text)
    
    if match:
        compression_ratio = float(match.group(1))
        return compression_ratio
    else:
        return None
 
def extract_compression_ratio(log_text):
    # Use regular expression to find all compression ratios
    compression_ratios = re.findall(r'compression ratio\s+=\s+([\d.]+)', log_text)
    return [float(ratio) for ratio in compression_ratios]
'''   
# Example usage:
log_filename = 'gzip_script_output.log'
n = 1  # Adjust n to the desired line interval

log_lines = read_log_file(log_filename, n)

if log_lines:
    compression_ratios = []
    for line in log_lines:
        ratio = extract_compression_ratio(line)
        if ratio is not None:
            compression_ratios.append(ratio)
            print(f"压缩比率: {ratio}")
    if not compression_ratios:
        print("未找到压缩比率信息。")
else:
    print(f"日志文件 '{log_filename}' 不存在。")
